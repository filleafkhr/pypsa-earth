# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Prepares brownfield data from previous planning horizon.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from add_existing_baseyear import (
    add_build_year_to_new_assets, apply_gas_trade_adjustments,check_and_fix_expansion_limits,fix_pypsa_consistency_warnings,
    filter_transmission_project_build_year,_assert_nom_bounds
)
from prepare_sector_network import _assert_no_nans_in_timeseries, _fill_nan_store_p_nom, _assert_component_bounds_sane
from _helpers import sanitize_carriers, sanitize_locations

# from pypsa.clustering.spatial import normed_or_uniform

logger = logging.getLogger(__name__)
idx = pd.IndexSlice

def normalize_key(k: str) -> str:
    return k.strip().lower().replace("_", " ").replace("-", " ")

def any_match(series, patterns):
    if series.empty:
        return series
    pat = "|".join([f"({p})" for p in patterns])
    return series.str.contains(pat, case=False, regex=True, na=False)


def _bus_country_map(n):
    if "country" in n.buses.columns:
        return n.buses["country"]
    return n.buses.index.to_series().str.extract(r"^([A-Z]{2})")[0].rename("country")

def _country_of(n, df, comp):
    buscol = "bus" if comp in ("Generator","FuelGenerator","Store") else "bus1"
    return _bus_country_map(n).reindex(df[buscol]).fillna("")


def apply_coal_supplier_phaseout(n, year, elec_cfg, per_bus_factors=None, verbose=True):

    def _cfg_fraction(y: int, cfg: dict) -> float:
        cp = (cfg or {}).get("coal_phaseout", {}) or {}
        if not cp.get("enable", False):
            return 1.0
        try:
            start = int(cp.get("start_year", y))
        except Exception:
            start = y

        raw = cp.get("target_year") or {}
        targets = {int(k): float(v) for k, v in raw.items()}
        if not targets:
            return 1.0

        if y <= start:
            return 1.0
        items = sorted(targets.items())
        if y >= items[-1][0]:
            return items[-1][1]

        y1, f1 = start, 1.0
        for y2, f2 in items:
            if y <= y2:
                t = (y - y1) / float(y2 - y1)
                return f1 + t * (f2 - f1)
            y1, f1 = y2, f2
        return f1

    def _coal_like_df(df):
        name = df["name"] if "name" in df else pd.Series("", index=df.index)
        carr = df["carrier"] if "carrier" in df else pd.Series("", index=df.index)
        patt = r"\b(?:coal|lignite)\b"
        return (
            name.astype(str).str.contains(patt, case=False, regex=True, na=False)
            | carr.astype(str).str.contains(patt, case=False, regex=True, na=False)
        )
    def _dbg_df(df, mask, label, topn=12):
        if not verbose or df is None or getattr(df, "empty", True):
            return
        m = mask.fillna(False) if isinstance(mask, pd.Series) else mask
        cnt = int(m.sum()) if hasattr(m, "sum") else 0
        print(f"[coal phaseout][dbg] {label}: affected={cnt} / {len(df)}")
        if cnt == 0:
            return
        sub = df.loc[m].copy()
        ext = sub.get("p_nom_extendable", False)
        ext_cnt = int(ext.fillna(False).sum()) if isinstance(ext, pd.Series) else 0
        print(f"[coal phaseout][dbg] {label}: extendable={ext_cnt}, fixed={cnt - ext_cnt}")
        for col in ["p_nom", "p_nom_min", "p_nom_max"]:
            if col in sub.columns:
                v = pd.to_numeric(sub[col], errors="coerce")
                n_nan = int(v.isna().sum())
                n_inf = int(np.isinf(v.to_numpy()).sum())
                print(f"[coal phaseout][dbg] {label}: {col} nan={n_nan}, inf={n_inf}")
        for col in ["p_nom", "p_nom_min", "p_nom_max"]:
            if col in sub.columns:
                v = pd.to_numeric(sub[col], errors="coerce").fillna(0.0)
                print(f"[coal phaseout][dbg] {label}: sum({col})={float(v.sum()):.6g}")
        sort_col = "p_nom" if "p_nom" in sub.columns else ("p_nom_max" if "p_nom_max" in sub.columns else None)
        if sort_col is not None:
            vv = pd.to_numeric(sub[sort_col], errors="coerce").fillna(0.0)
            idx = vv.sort_values(ascending=False).head(topn).index
            cols = [c for c in ["name", "carrier", "bus", "bus0", "bus1", "p_nom_extendable", "p_nom", "p_nom_min", "p_nom_max"] if c in sub.columns]
            print(f"[coal phaseout][dbg] {label}: top {min(topn, len(idx))} by {sort_col}:")
            print(sub.loc[idx, cols])

    def _scale_cap(df, mask, factor_like, label=""):
        if not getattr(mask, "any", lambda: False)():
            return

        if np.isscalar(factor_like):
            fac = pd.Series(float(factor_like), index=df.index)
        else:
            fac = pd.Series(factor_like).reindex(df.index).astype(float).fillna(1.0)

        ext = df.get("p_nom_extendable", False)
        ext = ext.fillna(False) if isinstance(ext, pd.Series) else pd.Series(False, index=df.index)

        zero_mask = mask & (fac <= 0.0)
        if zero_mask.any():
            if verbose:
                print(f"[coal phaseout][dbg] {label}: hard-zero rows={int(zero_mask.sum())}")
            for col in ["p_nom", "p_nom_min", "p_nom_max"]:
                if col in df.columns:
                    df.loc[zero_mask, col] = 0.0
            mask = mask & (~zero_mask)
            if not mask.any():
                if verbose:
                    _dbg_df(df, zero_mask, f"{label} (post hard-zero)")
                return

        ext_mask = mask & ext
        fix_mask = mask & (~ext)

        if verbose:
            _dbg_df(df, mask | zero_mask, f"{label} (pre-scale)")
            fsub = fac.loc[mask | zero_mask].astype(float)
            print(
                f"[coal phaseout][dbg] {label}: factor stats "
                f"min={float(fsub.min()):.4g}, mean={float(fsub.mean()):.4g}, max={float(fsub.max()):.4g}"
            )

        if fix_mask.any():
            fac_fix = fac.loc[fix_mask].astype(float)
            if "p_nom" in df:
                pnom = pd.to_numeric(df.loc[fix_mask, "p_nom"], errors="coerce").fillna(0.0)
                df.loc[fix_mask, "p_nom"] = pnom * fac_fix
            if "p_nom_min" in df:
                pmin = pd.to_numeric(df.loc[fix_mask, "p_nom_min"], errors="coerce").fillna(0.0)
                df.loc[fix_mask, "p_nom_min"] = pmin * fac_fix
            if "p_nom" in df and "p_nom_min" in df:
                pnom_new = pd.to_numeric(df.loc[fix_mask, "p_nom"], errors="coerce").fillna(0.0)
                pmin_new = pd.to_numeric(df.loc[fix_mask, "p_nom_min"], errors="coerce").fillna(0.0)
                df.loc[fix_mask, "p_nom_min"] = np.minimum(pmin_new, pnom_new)

        # extendable assets
        if ext_mask.any():
            fac_ext = fac.loc[ext_mask].astype(float)
            if "p_nom_max" in df:
                pmax_raw = pd.to_numeric(df.loc[ext_mask, "p_nom_max"], errors="coerce")
                if verbose:
                    print(
                        f"[coal phaseout][dbg] {label}: ext p_nom_max raw "
                        f"nan={int(pmax_raw.isna().sum())}, inf={int(np.isinf(pmax_raw.to_numpy()).sum())}"
                    )
                pmax = pmax_raw.fillna(np.inf)
                df.loc[ext_mask, "p_nom_max"] = pmax * fac_ext
            if "p_nom_min" in df:
                pmin = pd.to_numeric(df.loc[ext_mask, "p_nom_min"], errors="coerce").fillna(0.0)
                df.loc[ext_mask, "p_nom_min"] = pmin * fac_ext
            if "p_nom" in df:
                pnom = pd.to_numeric(df.loc[ext_mask, "p_nom"], errors="coerce").fillna(0.0)
                if "p_nom_max" in df:
                    pmax_new = pd.to_numeric(df.loc[ext_mask, "p_nom_max"], errors="coerce")
                    finite = np.isfinite(pmax_new.to_numpy())
                    pnom_vals = pnom.to_numpy()
                    pmax_vals = pmax_new.to_numpy()
                    pnom_vals[finite] = np.minimum(pnom_vals[finite], pmax_vals[finite])
                    df.loc[ext_mask, "p_nom"] = pnom_vals
                else:
                    df.loc[ext_mask, "p_nom"] = pnom

        if verbose:
            _dbg_df(df, (mask | zero_mask), f"{label} (post-scale)")
    def _scale_store_cap(df, mask, factor_like, label=""):
        """
        Scale Store energy capacities (e_nom/e_nom_min/e_nom_max) and e_initial.
        Mirrors _scale_cap logic but for Store energy, not power.
        """
        if not getattr(mask, "any", lambda: False)():
            return

        if np.isscalar(factor_like):
            fac = pd.Series(float(factor_like), index=df.index)
        else:
            fac = pd.Series(factor_like).reindex(df.index).astype(float).fillna(1.0)

        ext = df.get("e_nom_extendable", False)
        ext = ext.fillna(False) if isinstance(ext, pd.Series) else pd.Series(False, index=df.index)

        # hard-zero rows where factor <= 0 to avoid inf*0 -> NaN
        zero_mask = mask & (fac <= 0.0)
        if zero_mask.any():
            if verbose:
                print(f"[coal phaseout][dbg] {label}: hard-zero rows={int(zero_mask.sum())}")
            for col in ["e_nom", "e_nom_min", "e_nom_max", "e_initial"]:
                if col in df.columns:
                    df.loc[zero_mask, col] = 0.0
            mask = mask & (~zero_mask)
            if not mask.any():
                if verbose:
                    _dbg_df(df, zero_mask, f"{label} (post hard-zero)")
                return

        ext_mask = mask & ext
        fix_mask = mask & (~ext)

        if verbose:
            _dbg_df(df, mask | zero_mask, f"{label} (pre-scale)")
            fsub = fac.loc[mask | zero_mask].astype(float)
            print(
                f"[coal phaseout][dbg] {label}: factor stats "
                f"min={float(fsub.min()):.4g}, mean={float(fsub.mean()):.4g}, max={float(fsub.max()):.4g}"
            )

        # fixed stores: scale e_nom / e_nom_min and clamp min<=nom
        if fix_mask.any():
            fac_fix = fac.loc[fix_mask].astype(float)
            if "e_nom" in df:
                enom = pd.to_numeric(df.loc[fix_mask, "e_nom"], errors="coerce").fillna(0.0)
                df.loc[fix_mask, "e_nom"] = enom * fac_fix
            if "e_nom_min" in df:
                emin = pd.to_numeric(df.loc[fix_mask, "e_nom_min"], errors="coerce").fillna(0.0)
                df.loc[fix_mask, "e_nom_min"] = emin * fac_fix
            if "e_nom" in df and "e_nom_min" in df:
                enom_new = pd.to_numeric(df.loc[fix_mask, "e_nom"], errors="coerce").fillna(0.0)
                emin_new = pd.to_numeric(df.loc[fix_mask, "e_nom_min"], errors="coerce").fillna(0.0)
                df.loc[fix_mask, "e_nom_min"] = np.minimum(emin_new, enom_new)

        # extendable stores: scale e_nom_max / e_nom_min and clamp e_nom<=e_nom_max
        if ext_mask.any():
            fac_ext = fac.loc[ext_mask].astype(float)
            if "e_nom_max" in df:
                emax_raw = pd.to_numeric(df.loc[ext_mask, "e_nom_max"], errors="coerce")
                if verbose:
                    print(
                        f"[coal phaseout][dbg] {label}: ext e_nom_max raw "
                        f"nan={int(emax_raw.isna().sum())}, inf={int(np.isinf(emax_raw.to_numpy()).sum())}"
                    )
                emax = emax_raw.fillna(np.inf)
                df.loc[ext_mask, "e_nom_max"] = emax * fac_ext
            if "e_nom_min" in df:
                emin = pd.to_numeric(df.loc[ext_mask, "e_nom_min"], errors="coerce").fillna(0.0)
                df.loc[ext_mask, "e_nom_min"] = emin * fac_ext
            if "e_nom" in df:
                enom = pd.to_numeric(df.loc[ext_mask, "e_nom"], errors="coerce").fillna(0.0)
                if "e_nom_max" in df:
                    emax_new = pd.to_numeric(df.loc[ext_mask, "e_nom_max"], errors="coerce")
                    finite = np.isfinite(emax_new.to_numpy())
                    enom_vals = enom.to_numpy()
                    emax_vals = emax_new.to_numpy()
                    enom_vals[finite] = np.minimum(enom_vals[finite], emax_vals[finite])
                    df.loc[ext_mask, "e_nom"] = enom_vals
                else:
                    df.loc[ext_mask, "e_nom"] = enom

        # scale e_initial too
        if "e_initial" in df.columns:
            ei = pd.to_numeric(df.loc[mask, "e_initial"], errors="coerce").fillna(0.0)
            df.loc[mask, "e_initial"] = (ei * fac.loc[mask].astype(float)).values

        if verbose:
            _dbg_df(df, (mask | zero_mask), f"{label} (post-scale)")

    fraction = float(_cfg_fraction(int(year), elec_cfg))
    if verbose:
        cp = (elec_cfg or {}).get("coal_phaseout", {}) or {}
        print(
            f"[coal phaseout] year={year} fraction={fraction:.3f} | "
            f"cfg: enable={cp.get('enable', False)} start_year={cp.get('start_year', None)} "
            f"targets={cp.get('target_year', None)}"
        )

    per_bus_factors = per_bus_factors or {}


    if not n.loads.empty:

        coal_ind_mask = n.loads.index.astype(str).str.endswith(" coal for industry")
        if "carrier" in n.loads.columns:
            coal_ind_mask = coal_ind_mask | (n.loads["carrier"].astype(str) == "coal for industry")

        coal_ind_loads = n.loads.index[coal_ind_mask]

        if verbose:
            print(f"[coal phaseout][dbg] industry coal loads found: {len(coal_ind_loads)}")

        if len(coal_ind_loads):
            # infer corresponding gas load names: replace suffix
            gas_ind_loads = pd.Index(
                [str(i).replace(" coal for industry", " gas for industry") for i in coal_ind_loads],
                dtype=str,
            )

            missing_gas = gas_ind_loads.difference(n.loads.index)
            if len(missing_gas):
                if verbose:
                    print(f"[coal phaseout][dbg] creating missing gas loads: {len(missing_gas)}")
                for ld in missing_gas:
                    bus = ld
                    if bus not in n.buses.index:
                        if verbose:
                            print(f"[coal phaseout][warn] gas bus '{bus}' not in n.buses; cannot create Load '{ld}'")
                        continue
                    n.add("Load", ld, bus=bus)
                    if "carrier" in n.loads.columns:
                        n.loads.at[ld, "carrier"] = "gas for industry"

            gas_ind_loads = gas_ind_loads.intersection(n.loads.index)

            coal_pset = pd.to_numeric(n.loads.loc[coal_ind_loads, "p_set"], errors="coerce").fillna(0.0)

            if fraction <= 0.0:
                shift = 1.0
                if verbose:
                    print(f"[coal phaseout] fraction=0 -> shifting {len(coal_ind_loads)} coal industry loads fully to gas")
                if len(gas_ind_loads):
                    gas_now = pd.to_numeric(n.loads.loc[gas_ind_loads, "p_set"], errors="coerce").fillna(0.0)
                    coal_to_gas = pd.Series(gas_ind_loads, index=coal_ind_loads)
                    add_to_gas = coal_pset.copy()
                    add_to_gas.index = coal_to_gas.loc[add_to_gas.index].values
                    n.loads.loc[gas_ind_loads, "p_set"] = (gas_now + add_to_gas.reindex(gas_ind_loads).fillna(0.0)).values

                # zero coal loads
                n.loads.loc[coal_ind_loads, "p_set"] = 0.0

            elif 0.0 < fraction < 1.0:
                shift = 1.0 - fraction
                if verbose:
                    print(f"[coal phaseout] partial -> coal*{fraction:.3f}, shift {shift:.3f} to gas for industry")

                # scale coal down
                n.loads.loc[coal_ind_loads, "p_set"] = (coal_pset * fraction).values

                # add shifted part to gas
                if len(gas_ind_loads):
                    gas_now = pd.to_numeric(n.loads.loc[gas_ind_loads, "p_set"], errors="coerce").fillna(0.0)
                    coal_to_gas = pd.Series(gas_ind_loads, index=coal_ind_loads)
                    add_to_gas = (coal_pset * shift)
                    add_to_gas.index = coal_to_gas.loc[add_to_gas.index].values
                    n.loads.loc[gas_ind_loads, "p_set"] = (gas_now + add_to_gas.reindex(gas_ind_loads).fillna(0.0)).values
            if "industry coal emissions" in n.loads.index:

                coal_remaining = pd.to_numeric(n.loads.loc[coal_ind_loads, "p_set"], errors="coerce").fillna(0.0).sum()

                try:
                    coal_intensity = float(n.carriers.at["coal", "co2_emissions"])  
                except Exception:
                    coal_intensity = None

                if coal_intensity is not None:
                    n.loads.at["industry coal emissions", "p_set"] = -float(coal_remaining * coal_intensity)
                    if verbose:
                        print(f"[coal phaseout][dbg] updated 'industry coal emissions' to {-coal_remaining * coal_intensity:.6g}")
                else:
                    if verbose:
                        print("[coal phaseout][warn] could not infer coal CO2 intensity here; 'industry coal emissions' not updated.")

    coal_bus_mask = (
        n.buses["carrier"].astype(str).str.contains(r"\b(?:coal|lignite)\b", case=False, regex=True, na=False)
        if ("carrier" in n.buses)
        else pd.Series(False, index=n.buses.index)
    )
    coal_buses = n.buses.index[coal_bus_mask]

    if not n.generators.empty:
        g_mask = _coal_like_df(n.generators)
        if verbose:
            _dbg_df(n.generators, g_mask, "Generators (identified)")
        if g_mask.any():
            if per_bus_factors:
                g_fac = pd.Series(
                    [per_bus_factors.get(b, fraction) for b in n.generators["bus"]],
                    index=n.generators.index,
                    dtype=float,
                )
                _scale_cap(n.generators, g_mask, g_fac, label="Generators")
            else:
                _scale_cap(n.generators, g_mask, fraction, label="Generators")
  
    if hasattr(n, "stores") and not n.stores.empty:
        s_mask_name = _coal_like_df(n.stores)
        s_mask_bus = n.stores["bus"].isin(coal_buses) if "bus" in n.stores else pd.Series(False, index=n.stores.index)
        s_mask = s_mask_name | s_mask_bus

        if verbose:
            print(
                f"[coal phaseout][dbg] Stores masks: "
                f"name/carrier={int(s_mask_name.sum())}, bus_on_coal_bus={int(s_mask_bus.sum())}, union={int(s_mask.sum())}"
            )

        if s_mask.any():
            if per_bus_factors and "bus" in n.stores:
                s_fac = pd.Series(
                    [per_bus_factors.get(b, fraction) for b in n.stores["bus"]],
                    index=n.stores.index,
                    dtype=float,
                )
                _scale_store_cap(n.stores, s_mask, s_fac, label="Stores")
            else:
                _scale_store_cap(n.stores, s_mask, fraction, label="Stores")
                

    if not n.links.empty:
        l_mask_name = _coal_like_df(n.links)
        l_mask_bus0 = n.links["bus0"].isin(coal_buses) if "bus0" in n.links else pd.Series(False, index=n.links.index)
        l_mask = l_mask_name | l_mask_bus0

        if verbose:
            print(
                f"[coal phaseout][dbg] Links masks: "
                f"name/carrier={int(l_mask_name.sum())}, bus0_on_coal_bus={int(l_mask_bus0.sum())}, union={int(l_mask.sum())}"
            )

        if l_mask.any():
            if per_bus_factors and "bus0" in n.links:
                l_fac = pd.Series(
                    [per_bus_factors.get(b0, fraction) for b0 in n.links["bus0"]],
                    index=n.links.index,
                    dtype=float,
                )
                _scale_cap(n.links, l_mask, l_fac, label="Links")
            else:
                _scale_cap(n.links, l_mask, fraction, label="Links")

    if verbose:
        def _sum(df, sel):
            if df is None or getattr(df, "empty", True) or not sel.any() or "p_nom" not in df:
                return 0.0
            return float(pd.to_numeric(df.loc[sel, "p_nom"], errors="coerce").fillna(0.0).sum())

        g_sel = _coal_like_df(n.generators) if not n.generators.empty else pd.Series([], dtype=bool)
        l_sel = _coal_like_df(n.links) if not n.links.empty else pd.Series([], dtype=bool)
        g_cap = _sum(n.generators, g_sel)
        l_cap = _sum(n.links, l_sel)
        print(
            f"[coal phaseout] done | year={year} factor={fraction:.3f} | "
            f"[supplier-side] gen_p_nom_sum≈{g_cap:.6g} MW, link_p_nom_sum≈{l_cap:.6g} MW"
        )

    return fraction


def add_brownfield(n, n_p, year):
    logger.info(f"Preparing brownfield for the year {year}")

    # electric transmission grid set optimised capacities of previous as minimum
    n.lines.s_nom_min = n_p.lines.s_nom_opt
    dc_i = n.links[n.links.carrier == "DC"].index
    dc_common = dc_i.intersection(n_p.links.index)
    dc_missing = dc_i.difference(n_p.links.index)

    if len(dc_missing):
        logger.warning(
            f"brownfield: {len(dc_missing)} DC link(s) exist only in current horizon; "
            f"leaving them unchanged. example: {list(dc_missing)}"
    )

    # set p_nom_min only for those that exist in both years
    n.links.loc[dc_common, "p_nom_min"] = n_p.links.loc[dc_common, "p_nom_opt"]

    for c in n_p.iterate_components(["Link", "Generator", "Store"]):
        attr = "e" if c.name == "Store" else "p"

        # first, remove generators, links and stores that track
        # CO2 or global EU values since these are already in n
        n_p.mremove(c.name, c.df.index[c.df.lifetime == np.inf])

        # remove assets whose build_year + lifetime < year
        n_p.mremove(c.name, c.df.index[c.df.build_year + c.df.lifetime < year])

        # remove assets if their optimized nominal capacity is lower than a threshold
        # since CHP heat Link is proportional to CHP electric Link, make sure threshold is compatible
        chp_heat = c.df.index[
            (c.df[f"{attr}_nom_extendable"] & c.df.index.str.contains("urban central"))
            & c.df.index.str.contains("CHP")
            & c.df.index.str.contains("heat")
        ]

        threshold = snakemake.params.threshold_capacity

        if not chp_heat.empty:
            threshold_chp_heat = (
                threshold
                * c.df.efficiency[chp_heat.str.replace("heat", "electric")].values
                * c.df.p_nom_ratio[chp_heat.str.replace("heat", "electric")].values
                / c.df.efficiency[chp_heat].values
            )
            n_p.mremove(
                c.name,
                chp_heat[c.df.loc[chp_heat, f"{attr}_nom_opt"] < threshold_chp_heat],
            )

        n_p.mremove(
            c.name,
            c.df.index[
                (c.df[f"{attr}_nom_extendable"] & ~c.df.index.isin(chp_heat))
                & (c.df[f"{attr}_nom_opt"] < threshold)
            ],
        )

        # copy over assets but fix their capacity
        c.df[f"{attr}_nom"] = c.df[f"{attr}_nom_opt"]
        c.df[f"{attr}_nom_extendable"] = False

        n.import_components_from_dataframe(c.df, c.name)

        # copy time-dependent
        selection = n.component_attrs[c.name].type.str.contains(
            "series"
        ) & n.component_attrs[c.name].status.str.contains("Input")
        for tattr in n.component_attrs[c.name].index[selection]:
            n.import_series_from_dataframe(c.pnl[tattr], c.name, tattr)

        # deal with gas network
        pipe_carrier = ["gas pipeline"]
        if snakemake.params.H2_retrofit:
            # drop capacities of previous year to avoid duplicating
            to_drop = n.links.carrier.isin(pipe_carrier) & (n.links.build_year != year)
            n.mremove("Link", n.links.loc[to_drop].index)

            # subtract the already retrofitted from today's gas grid capacity
            h2_retrofitted_fixed_i = n.links[
                (n.links.carrier == "H2 pipeline retrofitted")
                & (n.links.build_year != year)
            ].index
            gas_pipes_i = n.links[n.links.carrier.isin(pipe_carrier)].index
            CH4_per_H2 = 1 / snakemake.params.H2_retrofit_capacity_per_CH4
            fr = "H2 pipeline retrofitted"
            to = "gas pipeline"
            # today's pipe capacity
            pipe_capacity = n.links.loc[gas_pipes_i, "p_nom"]
            # already retrofitted capacity from gas -> H2
            already_retrofitted = (
                n.links.loc[h2_retrofitted_fixed_i, "p_nom"]
                .rename(lambda x: x.split("-2")[0].replace(fr, to))
                .groupby(level=0)
                .sum()
            )
            remaining_capacity = (
                pipe_capacity
                - CH4_per_H2
                * already_retrofitted.reindex(index=pipe_capacity.index).fillna(0)
            )
            n.links.loc[gas_pipes_i, "p_nom"] = remaining_capacity
        else:
            new_pipes = n.links.carrier.isin(pipe_carrier) & (
                n.links.build_year == year
            )
            n.links.loc[new_pipes, "p_nom"] = 0.0
            n.links.loc[new_pipes, "p_nom_min"] = 0.0


def disable_grid_expansion_if_limit_hit(n):
    """
    Check if transmission expansion limit is already reached; then turn off.

    In particular, this function checks if the total transmission
    capital cost or volume implied by s_nom_min and p_nom_min are
    numerically close to the respective global limit set in
    n.global_constraints. If so, the nominal capacities are set to the
    minimum and extendable is turned off; the corresponding global
    constraint is then dropped.
    """
    cols = {"cost": "capital_cost", "volume": "length"}
    for limit_type in ["cost", "volume"]:
        glcs = n.global_constraints.query(
            f"type == 'transmission_expansion_{limit_type}_limit'"
        )

        for name, glc in glcs.iterrows():
            total_expansion = (
                (
                    n.lines.query("s_nom_extendable")
                    .eval(f"s_nom_min * {cols[limit_type]}")
                    .sum()
                )
                + (
                    n.links.query("carrier == 'DC' and p_nom_extendable")
                    .eval(f"p_nom_min * {cols[limit_type]}")
                    .sum()
                )
            ).sum()

            # Allow small numerical differences
            if np.abs(glc.constant - total_expansion) / glc.constant < 1e-6:
                logger.info(
                    f"Transmission expansion {limit_type} is already reached, disabling expansion and limit"
                )
                extendable_acs = n.lines.query("s_nom_extendable").index
                n.lines.loc[extendable_acs, "s_nom_extendable"] = False
                n.lines.loc[extendable_acs, "s_nom"] = n.lines.loc[
                    extendable_acs, "s_nom_min"
                ]

                extendable_dcs = n.links.query(
                    "carrier == 'DC' and p_nom_extendable"
                ).index
                n.links.loc[extendable_dcs, "p_nom_extendable"] = False
                n.links.loc[extendable_dcs, "p_nom"] = n.links.loc[
                    extendable_dcs, "p_nom_min"
                ]

                n.global_constraints.drop(name, inplace=True)


# def adjust_renewable_profiles(n, input_profiles, params, year):
#     """
#     Adjusts renewable profiles according to the renewable technology specified,
#     using the latest year below or equal to the selected year.
#     """

#     # spatial clustering
#     cluster_busmap = pd.read_csv(snakemake.input.cluster_busmap, index_col=0).squeeze()
#     simplify_busmap = pd.read_csv(
#         snakemake.input.simplify_busmap, index_col=0
#     ).squeeze()
#     clustermaps = simplify_busmap.map(cluster_busmap)
#     clustermaps.index = clustermaps.index.astype(str)

#     # temporal clustering
#     dr = pd.date_range(**params["snapshots"], freq="h")
#     snapshotmaps = (
#         pd.Series(dr, index=dr).where(lambda x: x.isin(n.snapshots), pd.NA).ffill()
#     )

#     for carrier in params["carriers"]:
#         if carrier == "hydro":
#             continue
#         with xr.open_dataset(getattr(input_profiles, "profile_" + carrier)) as ds:
#             if ds.indexes["bus"].empty or "year" not in ds.indexes:
#                 continue

#             closest_year = max(
#                 (y for y in ds.year.values if y <= year), default=min(ds.year.values)
#             )

#             p_max_pu = (
#                 ds["profile"]
#                 .sel(year=closest_year)
#                 .transpose("time", "bus")
#                 .to_pandas()
#             )

#             # spatial clustering
#             weight = ds["weight"].sel(year=closest_year).to_pandas()
#             weight = weight.groupby(clustermaps).transform(normed_or_uniform)
#             p_max_pu = (p_max_pu * weight).T.groupby(clustermaps).sum().T
#             p_max_pu.columns = p_max_pu.columns + f" {carrier}"

#             # temporal_clustering
#             p_max_pu = p_max_pu.groupby(snapshotmaps).mean()

#             # replace renewable time series
#             n.generators_t.p_max_pu.loc[:, p_max_pu.columns] = p_max_pu



    def _fix_nan_p_nom_max(n):
        for df_name in ["generators", "links"]:
            df = getattr(n, df_name)
            if df.empty or "p_nom_max" not in df.columns:
                continue

            pmax = pd.to_numeric(df["p_nom_max"], errors="coerce")
            ext = df.get("p_nom_extendable", False)
            ext = ext.fillna(False).astype(bool) if isinstance(ext, pd.Series) else pd.Series(False, index=df.index)

            nan = pmax.isna()
            if nan.any():
                # only meaningful for extendables; set to inf
                fix = nan & ext
                if fix.any():
                    df.loc[fix, "p_nom_max"] = np.inf
                    print(f"[fix] set {fix.sum()} NaN {df_name}.p_nom_max to inf (extendable assets)")


if __name__ == "__main__":
    if "snakemake" not in globals():

        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_brownfield",
            simpl="",
            clusters="4",
            ll="c1",
            opts="Co2L-4H",
            planning_horizons="2030",
            sopts="144H",
            discountrate=0.071,
            demand="AB",
            h2export="120",
        )

    logger.info(f"Preparing brownfield from the file {snakemake.input.network_p}")

    year = int(snakemake.wildcards.planning_horizons)

    n = pypsa.Network(snakemake.input.network)

    # TODO
    # adjust_renewable_profiles(n, snakemake.input, snakemake.params, year)

    add_build_year_to_new_assets(n, year)

    n_p = pypsa.Network(snakemake.input.network_p)
    add_brownfield(n, n_p, year)
    disable_grid_expansion_if_limit_hit(n)
    check_and_fix_expansion_limits(n)
    n = fix_pypsa_consistency_warnings(n, verbose=True) 
    elec_cfg = snakemake.config.get("electricity", {})
    apply_coal_supplier_phaseout(n, year, elec_cfg, verbose=True)
    apply_gas_trade_adjustments(n, snakemake.config)
    sanitize_carriers(n, snakemake.config)
    sanitize_locations(n)
    _assert_nom_bounds(n, tag=f"after coal phaseout {year}")

    _fix_nan_p_nom_max(n)
    _fill_nan_store_p_nom(n, hours_default=1.0, pnom_floor=1.0, make_extendable=True)

    _assert_no_nans_in_timeseries(n)
    _assert_component_bounds_sane(n)
    print("[debug] sanity checks passed: no NaNs/infs, bounds look sane")

    filter_transmission_project_build_year(
        n,
        snakemake.params.transmission_projects,
        year,
    )

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])