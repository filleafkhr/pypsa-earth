import os

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pypsa
import scipy as sp
import shapely.prepared
import shapely.wkt
from _helpers import configure_logging, create_logger, read_csv_nafix
from shapely.ops import unary_union

logger = create_logger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("base_cleanup")

    configure_logging(snakemake)

    inputs = snakemake.input

    # Snakemake imports:
    base_network_config = snakemake.params.base_network

    n = pypsa.Network(snakemake.input.network)

    params = snakemake.params

    if base_network_config.get("cleanup", False):

        raw_whitelist = base_network_config.get("cleanup_whitelist") or []
        whitelist_pairs = {tuple(sorted(p)) for p in raw_whitelist}

        def _recompute_country_cols(n, tbl: str, canonize_for_clustering: bool = True):
            df = getattr(n, tbl)
            if df.empty:
                return
            b_country = n.buses["country"]
            df = df.copy()

            # recompute bus-derived countries (for traceability)
            df["bus_country0"] = b_country.reindex(df["bus0"]).values
            df["bus_country1"] = b_country.reindex(df["bus1"]).values

            if canonize_for_clustering:
                c0 = df["bus_country0"].astype(str).values
                c1 = df["bus_country1"].astype(str).values
                df["country0"] = np.minimum(c0, c1)
                df["country1"] = np.maximum(c0, c1)
            else:
                df["country0"] = df["bus_country0"]
                df["country1"] = df["bus_country1"]

            setattr(n, tbl, df)

        def _log_capacity_changes(table: str, df_before: pd.DataFrame, df_after: pd.DataFrame, cap_col: str, context_note: str = ""):
            """Summarize what changed, print a few examples."""
            a = df_after[cap_col].astype(float)
            b = df_before[cap_col].astype(float)
            delta = (a - b).fillna(0.0)
            changed_mask = delta.abs() > 1e-9
            num_changed = int(changed_mask.sum())
            total = int(len(delta))
            if num_changed == 0:
                logger.info(f"{table}: no {cap_col} changes {context_note}".strip())
                return

            abs_sum = float(delta.abs().sum())
            min_d = float(delta.min())
            max_d = float(delta.max())
            logger.info(
                f"{table}: changed {num_changed}/{total} rows in '{cap_col}' "
                f"(Σ|Δ|={abs_sum:.3f}; minΔ={min_d:.3f}; maxΔ={max_d:.3f}) {context_note}".strip()
            )

            # pick top 5 by absolute delta manually
            abs_delta = delta.abs()
            top_idx = abs_delta.sort_values(ascending=False).head(5).index

            cols_show = [c for c in ["carrier","bus0","bus1","bus_country0","bus_country1","country0","country1"] if c in df_after.columns]
            sample = pd.DataFrame({
                "old": b.loc[top_idx],
                "new": a.loc[top_idx],
                "Δ":   delta.loc[top_idx],
            }).join(df_after.loc[top_idx, cols_show])
            logger.info(f"{table}: sample changes:\n{sample.to_string()}")


        def _scale_table(
            n,
            table: str,
            cap_col: str,
            by_length_col: str = "length",
            cap_selector: str = "max",   # {"max","p95"}
            s_max_pu_default: float = 1.0,
            whitelist_pairs: set | None = None,
        ):
            df0 = getattr(n, table)
            if df0.empty:
                return

            # recompute countries; canonicalize for consistent grouping
            _recompute_country_cols(n, table, canonize_for_clustering=True)
            df = getattr(n, table).copy()

            if cap_col not in df.columns:
                logger.info(f"{table}: '{cap_col}' not found; skipping scaling.")
                return
            df[cap_col] = pd.to_numeric(df[cap_col], errors="coerce")

            # only cross-border (by canonical countries)
            cross = df["country0"].notna() & df["country1"].notna() & (df["country0"] != df["country1"])
            if not cross.any():
                logger.info(f"{table}: no cross-border assets to normalize.")
                setattr(n, table, df)
                return

            work = df.loc[cross].copy()

            # group by (carrier, canonical countries)
            if "carrier" not in work.columns:
                work["carrier"] = "AC" if table == "lines" else "DC"
            grp_cols = ["carrier", "country0", "country1"]

            # whitelist skipping
            if whitelist_pairs:
                work["skip"] = [(a, b) in whitelist_pairs for a, b in zip(work["country0"], work["country1"])]
            else:
                work["skip"] = False

            # corridor-level selected cap
            def _select_cap(s: pd.Series) -> float:
                vals = pd.to_numeric(s, errors="coerce").dropna().values
                if vals.size == 0:
                    return np.nan
                return float(np.percentile(vals, 95)) if cap_selector == "p95" else float(np.max(vals))

            work["corr_cap_sel"] = work.groupby(grp_cols)[cap_col].transform(_select_cap)

            # physics upper bound for AC lines if available — buses untouched
            work["phys_cap_corr"] = np.nan
            if table == "lines":
                if "s_max_pu" not in work.columns:
                    work["s_max_pu"] = s_max_pu_default
                if ("v_nom" in work.columns) and ("i_nom" in work.columns):
                    mask_vi = work["v_nom"].notna() & work["i_nom"].notna()
                    work.loc[mask_vi, "s_cap_seg"] = np.sqrt(3.0) * work.loc[mask_vi, "v_nom"] * work.loc[mask_vi, "i_nom"] * work.loc[mask_vi, "s_max_pu"]
                    work["phys_cap_corr"] = work.groupby(grp_cols)["s_cap_seg"].transform("max")

            # final corridor cap (respect physics & whitelist)
            corr_cap = work["corr_cap_sel"]
            if table == "lines":
                use_phys = work["phys_cap_corr"].notna()
                corr_cap = corr_cap.where(~use_phys, np.minimum(work["corr_cap_sel"], work["phys_cap_corr"]))
            corr_cap = corr_cap.where(work["skip"] == False, work[cap_col])
            work["corr_cap"] = corr_cap

            # weights by length (or equal if missing)
            if by_length_col not in work.columns:
                work[by_length_col] = 1.0
            len_sum = work.groupby(grp_cols)[by_length_col].transform("sum")
            cnt = work.groupby(grp_cols)[by_length_col].transform("count")
            with np.errstate(invalid="ignore", divide="ignore"):
                weights = np.where(len_sum > 0.0, work[by_length_col] / len_sum, 1.0 / cnt)
            work["weights"] = weights.astype(float)

            # compute new capacity, but don't write yet
            proposed = (work["corr_cap"] * work["weights"]).where(~work["skip"], work[cap_col]).astype(float)

            # ---- LOG DIFFS (before write) ----
            df_before = df.copy()
            df_after = df.copy()
            df_after.loc[work.index, cap_col] = proposed.values
            _log_capacity_changes(table, df_before, df_after, cap_col, context_note="(cleanup redistribution)")

            # write back
            df.loc[work.index, cap_col] = proposed.values
            setattr(n, table, df)

        def scale_crossborder_caps_by_length(n, cap_selector="max", whitelist_pairs=None):
            _scale_table(n, "lines", cap_col="s_nom", cap_selector=cap_selector, whitelist_pairs=whitelist_pairs)
            _scale_table(n, "links", cap_col="p_nom", cap_selector=cap_selector, whitelist_pairs=whitelist_pairs)

        # run cleanup (capacity redistribution only)
        scale_crossborder_caps_by_length(n, cap_selector="max", whitelist_pairs=whitelist_pairs)

        # refresh canonical countries once more for downstream clustering; buses remain as they were
        _recompute_country_cols(n, "lines", canonize_for_clustering=True)
        _recompute_country_cols(n, "links", canonize_for_clustering=True)

    n.export_to_netcdf(snakemake.output[0])