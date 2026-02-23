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
    add_build_year_to_new_assets,
    filter_transmission_project_build_year,lock_oil_electricity_assets,_assert_nom_bounds
)
from prepare_sector_network import _assert_no_nans_in_timeseries, _assert_component_bounds_sane
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

def apply_storage_country_rules(n, cfg):
    """
    Applies enable/disable and cost overrides for storage-related assets.

    YAML example:
    -------------
    feature_switches:
    storage_country_rules: true

    storage_rules:
    - country: "PH"
        enable: false
        carriers: ["battery","battery charger","battery discharger"]

    - country: "ID"
        enable: true
        carriers: ["battery"]
        mult: {"capital_cost": 0.95}

    - country: "*"
        enable: true
        carriers: ["home battery"]
    """
    if not cfg.get("feature_switches", {}).get("storage_country_rules", False):
        print("⚙️  storage_country_rules: OFF")
        return

    rules = cfg.get("storage_rules", [])
    if not rules:
        print("⚙️  storage_country_rules: no rules found")
        return

    # component tables and their country series
    G, L, S = n.generators, n.links, n.stores
    g_country = _country_of(n, G, "Generator") if not G.empty else pd.Series([], index=G.index, dtype=str)
    l_country = _country_of(n, L, "Link")      if not L.empty else pd.Series([], index=L.index, dtype=str)
    s_country = _country_of(n, S, "Store")     if not S.empty else pd.Series([], index=S.index, dtype=str)

    def build_mask(df, country_series, country_filter, carriers):
        if df.empty:
            return pd.Series(False, index=df.index)

        if carriers == ["*"]:
            mask_carrier = pd.Series(True, index=df.index)
        else:
            carriers_lower = {c.lower() for c in carriers}
            mask_carrier = df["carrier"].str.lower().isin(carriers_lower)

        if isinstance(country_filter, str):
            if country_filter == "*":
                mask_country = pd.Series(True, index=df.index)
            else:
                mask_country = (country_series == country_filter)
        elif isinstance(country_filter, (list, tuple, set)):
            mask_country = country_series.isin(list(country_filter))
        else:
            mask_country = pd.Series(False, index=df.index)

        return mask_carrier & mask_country

    def apply_rule_to_df(df, country_series, rule):
        if df.empty:
            return
        country_filter = rule.get("country", "*")
        carriers       = rule.get("carriers", ["*"])
        enable         = rule.get("enable", None)
        fields         = rule.get("fields", {}) or {}
        mult           = rule.get("mult", {}) or {}

        m = build_mask(df, country_series, country_filter, carriers)
        if not m.any():
            return

        if enable is not None:
            enable_bool = bool(enable)
            if "p_nom_extendable" in df.columns:
                df.loc[m, "p_nom_extendable"] = enable_bool
                if "p_nom_max" in df.columns:
                    df.loc[m, "p_nom_max"] = np.inf if enable_bool else df.loc[m, "p_nom"]

            if "e_nom_extendable" in df.columns:
                df.loc[m, "e_nom_extendable"] = enable_bool
                if "e_nom_max" in df.columns:
                    df.loc[m, "e_nom_max"] = np.inf if enable_bool else df.loc[m, "e_nom"]

        for k, fac in mult.items():
            if k in df.columns:
                df.loc[m, k] = df.loc[m, k] * float(fac)

        for k, val in fields.items():
            if k in df.columns:
                df.loc[m, k] = val

    for r in rules:
        apply_rule_to_df(G, g_country, r)
        apply_rule_to_df(L, l_country, r)
        apply_rule_to_df(S, s_country, r)

    print(f"applied {len(rules)} storage_country_rules")


def apply_gas_trade_adjustments(n, cfg):
    if not cfg.get("feature_switches", {}).get("gas_trade_adjustments", False):
        print("gas_trade_adjustments: OFF")
        return

    trade = cfg.get("gas_trade", {})
    importers = list(trade.get("importers", []))
    exporters = list(trade.get("exporters", []))
    imp_adj = trade.get("importer_adjustment", {}) or {}
    exp_adj = trade.get("exporter_adjustment", {}) or {}

    g = n.generators
    if not g.empty and "carrier" in g.columns:
        g_country = _country_of(n, g, "FuelGenerator")
        is_gas_fuel = g.carrier.str.lower().eq("gas")

        if importers and "fuel_marginal_cost_add" in imp_adj:
            m = is_gas_fuel & g_country.isin(importers)
            if m.any(): g.loc[m, "marginal_cost"] += float(imp_adj["fuel_marginal_cost_add"])

        if exporters and "fuel_marginal_cost_add" in exp_adj:
            m = is_gas_fuel & g_country.isin(exporters)
            if m.any(): g.loc[m, "marginal_cost"] += float(exp_adj["fuel_marginal_cost_add"])

    L = n.links
    if L.empty: return
    l_country = _country_of(n, L, "Link")
    is_ccgt = L.carrier.str.lower().eq("ccgt")
    is_ocgt = L.carrier.str.lower().eq("ocgt")

    def _apply_link_adjust(countries, adj):
        if not countries or not adj: return
        mask_country = l_country.isin(list(countries))

        if "ccgt_vom_add" in adj:
            m = mask_country & is_ccgt
            if m.any(): L.loc[m, "marginal_cost"] += float(adj["ccgt_vom_add"])
        if "ocgt_vom_add" in adj:
            m = mask_country & is_ocgt
            if m.any(): L.loc[m, "marginal_cost"] += float(adj["ocgt_vom_add"])

        if "ccgt_capex_mult" in adj:
            m = mask_country & is_ccgt
            if m.any(): L.loc[m, "capital_cost"] *= float(adj["ccgt_capex_mult"])
        if "ocgt_capex_mult" in adj:
            m = mask_country & is_ocgt
            if m.any(): L.loc[m, "capital_cost"] *= float(adj["ocgt_capex_mult"])

    _apply_link_adjust(importers, imp_adj)
    _apply_link_adjust(exporters, exp_adj)

    print("applied gas_trade_adjustments")

def apply_coal_supplier_phaseout(n, year, elec_cfg, per_bus_factors=None, verbose=True):
    """
    Coal/lignite phaseout with debug prints.

    NEW (industry coal -> gas switching):
    - Specifically shift *industry coal demand* (Loads named "... coal for industry") into the
      existing *gas for industry* Loads (named "... gas for industry"), instead of generic bus renaming.
    - Works for full (fraction=0) and partial (0<f<1) phaseout.
    - Updates the "industry coal emissions" Load to match the remaining coal demand.
      (Shifted demand becomes gas demand, so gas CO2 is handled by your gas->co2 Link.)
    - Keeps your supplier-side scaling for coal generators/links as before.
    """
    # ---- helpers (no extra imports) ----
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

    # ---- small debug helper ----
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

        # hard-zero rows where factor <= 0 to avoid inf*0 -> NaN
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

        # fixed assets
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

    # ---- compute fraction from config ----
    fraction = float(_cfg_fraction(int(year), elec_cfg))
    if verbose:
        cp = (elec_cfg or {}).get("coal_phaseout", {}) or {}
        print(
            f"[coal phaseout] year={year} fraction={fraction:.3f} | "
            f"cfg: enable={cp.get('enable', False)} start_year={cp.get('start_year', None)} "
            f"targets={cp.get('target_year', None)}"
        )

    per_bus_factors = per_bus_factors or {}

    # ======================================================================
    # INDUSTRY: COAL -> GAS demand switching (your specific Loads)
    # ======================================================================
    if not n.loads.empty:
        # coal industry loads created like: node + " coal for industry"
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

            # ensure gas loads exist (if your add_industry created them, they should exist)
            missing_gas = gas_ind_loads.difference(n.loads.index)
            if len(missing_gas):
                if verbose:
                    print(f"[coal phaseout][dbg] creating missing gas loads: {len(missing_gas)}")
                for ld in missing_gas:
                    # best guess bus name equals load name in your pattern
                    bus = ld
                    if bus not in n.buses.index:
                        # fallback: if bus not exist, skip rather than creating wrong topology
                        if verbose:
                            print(f"[coal phaseout][warn] gas bus '{bus}' not in n.buses; cannot create Load '{ld}'")
                        continue
                    n.add("Load", ld, bus=bus)
                    if "carrier" in n.loads.columns:
                        n.loads.at[ld, "carrier"] = "gas for industry"

            # align to existing gas loads only
            gas_ind_loads = gas_ind_loads.intersection(n.loads.index)

            # read current coal p_set (MW)
            coal_pset = pd.to_numeric(n.loads.loc[coal_ind_loads, "p_set"], errors="coerce").fillna(0.0)

            if fraction <= 0.0:
                # move 100% coal demand to gas
                shift = 1.0
                if verbose:
                    print(f"[coal phaseout] fraction=0 -> shifting {len(coal_ind_loads)} coal industry loads fully to gas")

                # add to gas loads
                if len(gas_ind_loads):
                    gas_now = pd.to_numeric(n.loads.loc[gas_ind_loads, "p_set"], errors="coerce").fillna(0.0)
                    # map by matching name replacement
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

            # fraction == 1 -> do nothing

            # Update coal emissions load to match remaining coal industry demand
            # (Shifted share is now gas and should emit via your gas->co2 link.)
            if "industry coal emissions" in n.loads.index:
                # remaining coal MW after scaling
                coal_remaining = pd.to_numeric(n.loads.loc[coal_ind_loads, "p_set"], errors="coerce").fillna(0.0).sum()
                # CO2 intensity for coal must be available in n.costs or costs table; here we assume you stored it on n
                # If you don't have access to `costs` here, read from n.links/n.carriers or pass it in.
                # Common pattern: put CO2 intensity into n.carriers or n.global_constraints externally.
                try:
                    coal_intensity = float(n.carriers.at["coal", "co2_emissions"])  # <- only if you have this
                except Exception:
                    # fallback: keep previous value if we can't infer intensity here
                    coal_intensity = None

                if coal_intensity is not None:
                    n.loads.at["industry coal emissions", "p_set"] = -float(coal_remaining * coal_intensity)
                    if verbose:
                        print(f"[coal phaseout][dbg] updated 'industry coal emissions' to {-coal_remaining * coal_intensity:.6g}")
                else:
                    if verbose:
                        print("[coal phaseout][warn] could not infer coal CO2 intensity here; 'industry coal emissions' not updated.")

    # ======================================================================
    # SUPPLIER-SIDE scaling (your existing logic)
    # ======================================================================

    # ---- identify coal buses (supplier side for links) ----
    coal_bus_mask = (
        n.buses["carrier"].astype(str).str.contains(r"\b(?:coal|lignite)\b", case=False, regex=True, na=False)
        if ("carrier" in n.buses)
        else pd.Series(False, index=n.buses.index)
    )
    coal_buses = n.buses.index[coal_bus_mask]
    if verbose:
        print(f"[coal phaseout][dbg] coal buses: {len(coal_buses)}")

    # ---- GENERATORS ----
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

    # ---- LINKS ----
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

def continuity_merge_after_brownfield(
    n,
    year: int,
    *,
    merge_generators=True,
    merge_links=True,
    gen_key_cols=("carrier", "bus"),
    link_key_cols=("carrier", "bus0", "bus1"),
    generator_carriers=None,   # optional whitelist (case-insensitive)
    link_carriers=None,        # optional whitelist (case-insensitive)
    set_min_to="p_nom",        # "p_nom" or "max(p_nom_min,p_nom)"
    verbose=True,
):
    """
    Continuity merge AFTER brownfield import.

    Goals:
      1) Remove year-template duplicates that represent the same physical asset:
         - Generators: key=(carrier,bus)
         - Links:      key=(carrier,bus0,bus1)
         template := build_year == year
         inherited := build_year < year

      2) Enforce continuity for inherited assets ONLY:
         - set p_nom_min to carry previous-year capacity floor
         - DO NOT change p_nom_extendable / p_nom_max / costs / efficiencies.

    This avoids the 'oil/ror unlocked' problem: no asset is made extendable here.
    """

    def _norm(x):
        return x.astype(str).str.strip().str.lower()

    def _norm_set(vals):
        if vals is None:
            return None
        return {str(v).strip().lower() for v in vals}

    gen_whitelist = _norm_set(generator_carriers)
    link_whitelist = _norm_set(link_carriers)

    def _make_key(df, cols):
        cols = [c for c in cols if c in df.columns]
        tmp = df.loc[:, cols].copy()
        if "carrier" in cols:
            tmp["carrier"] = _norm(tmp["carrier"])
        for c in cols:
            if c != "carrier":
                tmp[c] = tmp[c].fillna("").astype(str)
        return pd.Series(list(map(tuple, tmp.values)), index=df.index)

    def _apply_min_floor(df, idx):
        if idx.empty:
            return 0
        if "p_nom" not in df.columns or "p_nom_min" not in df.columns:
            return 0

        if set_min_to == "p_nom":
            df.loc[idx, "p_nom_min"] = df.loc[idx, "p_nom"]
        elif set_min_to == "max(p_nom_min,p_nom)":
            df.loc[idx, "p_nom_min"] = np.maximum(df.loc[idx, "p_nom_min"], df.loc[idx, "p_nom"])
        else:
            raise ValueError("set_min_to must be 'p_nom' or 'max(p_nom_min,p_nom)'")
        return int(len(idx))

    out = {
        "gen_dropped": 0,
        "gen_min_set": 0,
        "link_dropped": 0,
        "link_min_set": 0,
    }

    # ---------------- GENERATORS ----------------
    if merge_generators and not n.generators.empty:
        G = n.generators

        required = {"build_year", "carrier", "bus"}
        if not required.issubset(G.columns):
            if verbose:
                print(f"[continuity_merge] generators missing {required - set(G.columns)}, skipping generators")
        else:
            mask = pd.Series(True, index=G.index)

            if gen_whitelist is not None:
                mask &= _norm(G["carrier"]).isin(gen_whitelist)

            gen = G.loc[mask]
            inherited = gen.index[gen["build_year"] < year]
            template  = gen.index[gen["build_year"] == year]

            # drop template duplicates (same physical key as any inherited)
            if len(inherited) and len(template):
                key_inh = set(_make_key(G.loc[inherited], gen_key_cols).values)
                key_tpl = _make_key(G.loc[template], gen_key_cols)
                to_drop = key_tpl.index[key_tpl.isin(key_inh)]
                if len(to_drop):
                    n.mremove("Generator", to_drop)
                    out["gen_dropped"] = int(len(to_drop))

            # re-fetch after removals
            G = n.generators

            # IMPORTANT: recompute inherited index against updated table
            # (some inherited might have been removed upstream)
            inherited = G.index[(G["build_year"] < year)]
            if gen_whitelist is not None:
                inherited = inherited[_norm(G.loc[inherited, "carrier"]).isin(gen_whitelist)]

            out["gen_min_set"] = _apply_min_floor(G, inherited)

    # ---------------- LINKS ----------------
    if merge_links and not n.links.empty:
        L = n.links

        required = {"build_year", "carrier", "bus0", "bus1"}
        if not required.issubset(L.columns):
            if verbose:
                print(f"[continuity_merge] links missing {required - set(L.columns)}, skipping links")
        else:
            mask = pd.Series(True, index=L.index)
            if link_whitelist is not None:
                mask &= _norm(L["carrier"]).isin(link_whitelist)

            links = L.loc[mask]
            inherited = links.index[links["build_year"] < year]
            template  = links.index[links["build_year"] == year]

            if len(inherited) and len(template):
                key_inh = set(_make_key(L.loc[inherited], link_key_cols).values)
                key_tpl = _make_key(L.loc[template], link_key_cols)
                to_drop = key_tpl.index[key_tpl.isin(key_inh)]
                if len(to_drop):
                    n.mremove("Link", to_drop)
                    out["link_dropped"] = int(len(to_drop))

            # re-fetch after removals
            L = n.links

            inherited = L.index[(L["build_year"] < year)]
            if link_whitelist is not None:
                inherited = inherited[_norm(L.loc[inherited, "carrier"]).isin(link_whitelist)]

            out["link_min_set"] = _apply_min_floor(L, inherited)

    if verbose:
        print(
            f"[continuity_merge] year={year} | "
            f"gen: dropped={out['gen_dropped']}, p_nom_min_set={out['gen_min_set']} | "
            f"link: dropped={out['link_dropped']}, p_nom_min_set={out['link_min_set']}"
        )

    return out


def _norm_series(s):
    return s.astype(str).str.strip().str.lower()

def audit_after_continuity_merge(n, year, carriers=("oil", "ror", "run-of-river", "run of river")):
    """
    Prints:
      - summary by carrier (counts, p_nom sums, p_nom_min sums, extendable share)
      - template-vs-inherited duplicate keys that SURVIVED
      - inherited duplicates among themselves
    """
    carriers_norm = {c.strip().lower() for c in carriers}

    def _make_key(df, cols):
        cols = [c for c in cols if c in df.columns]
        tmp = df.loc[:, cols].copy()
        if "carrier" in cols:
            tmp["carrier"] = _norm_series(tmp["carrier"])
        for c in cols:
            if c != "carrier":
                tmp[c] = tmp[c].fillna("").astype(str)
        return pd.Series(list(map(tuple, tmp.values)), index=df.index)

    # ----------------- GENERATORS -----------------
    if not n.generators.empty and {"carrier","bus","build_year"}.issubset(n.generators.columns):
        G = n.generators.copy()
        G["carrier_norm"] = _norm_series(G["carrier"])

        # keep anything whose carrier contains "oil" or "ror" etc (substring helps catch "run-of-river")
        mask = G["carrier_norm"].apply(lambda x: any(k in x for k in carriers_norm))
        g = G.loc[mask].copy()

        print("\n=== generators: oil/ror audit ===")
        if g.empty:
            print("no matching generator carriers found")
        else:
            # summary by carrier
            def _colsum(df, col):
                return df[col].sum() if col in df.columns else np.nan

            summ = (g.groupby("carrier_norm")
                      .apply(lambda df: pd.Series({
                          "n": len(df),
                          "sum_p_nom": _colsum(df, "p_nom"),
                          "sum_p_nom_min": _colsum(df, "p_nom_min"),
                          "share_extendable": float(df["p_nom_extendable"].mean()) if "p_nom_extendable" in df else np.nan,
                          "min_build_year": df["build_year"].min(),
                          "max_build_year": df["build_year"].max(),
                      }))
                      .sort_values("sum_p_nom", ascending=False))
            print(summ)

            # template-vs-inherited duplicates that survived
            key = _make_key(g, ("carrier", "bus"))
            inh = g.index[g["build_year"] < year]
            tpl = g.index[g["build_year"] == year]
            if len(inh) and len(tpl):
                key_inh = set(key.loc[inh].values)
                surv_tpl = key.loc[tpl]
                survived = surv_tpl.index[surv_tpl.isin(key_inh)]
                print(f"\n[generators] template duplicates surviving merge: {len(survived)}")
                if len(survived):
                    show = g.loc[survived, ["carrier","bus","build_year","p_nom","p_nom_min","p_nom_extendable"]].copy()
                    print(show.sort_values(["carrier","bus"]).head(40))

            # inherited duplicates among themselves (merge won't fix these)
            if len(inh):
                inh_key = key.loc[inh]
                dup_inh = inh_key[inh_key.duplicated(keep=False)]
                print(f"[generators] inherited duplicates among themselves: {dup_inh.index.nunique()}")
                if not dup_inh.empty:
                    show = g.loc[dup_inh.index, ["carrier","bus","build_year","p_nom","p_nom_min","p_nom_extendable"]]
                    print(show.sort_values(["carrier","bus","build_year"]).head(60))

    else:
        print("\n=== generators: skipped (missing columns carrier/bus/build_year or empty) ===")

    # ----------------- LINKS (oil sometimes is here as conversion tech) -----------------
    if not n.links.empty and {"carrier","bus0","bus1","build_year"}.issubset(n.links.columns):
        L = n.links.copy()
        L["carrier_norm"] = _norm_series(L["carrier"])

        mask = L["carrier_norm"].apply(lambda x: any(k in x for k in carriers_norm))
        l = L.loc[mask].copy()

        print("\n=== links: oil/ror audit ===")
        if l.empty:
            print("no matching link carriers found")
        else:
            def _colsum(df, col):
                return df[col].sum() if col in df.columns else np.nan

            summ = (l.groupby("carrier_norm")
                      .apply(lambda df: pd.Series({
                          "n": len(df),
                          "sum_p_nom": _colsum(df, "p_nom"),
                          "sum_p_nom_min": _colsum(df, "p_nom_min"),
                          "share_extendable": float(df["p_nom_extendable"].mean()) if "p_nom_extendable" in df else np.nan,
                          "min_build_year": df["build_year"].min(),
                          "max_build_year": df["build_year"].max(),
                      }))
                      .sort_values("sum_p_nom", ascending=False))
            print(summ)

            key = _make_key(l, ("carrier", "bus0", "bus1"))
            inh = l.index[l["build_year"] < year]
            tpl = l.index[l["build_year"] == year]
            if len(inh) and len(tpl):
                key_inh = set(key.loc[inh].values)
                surv_tpl = key.loc[tpl]
                survived = surv_tpl.index[surv_tpl.isin(key_inh)]
                print(f"\n[links] template duplicates surviving merge: {len(survived)}")
                if len(survived):
                    show = l.loc[survived, ["carrier","bus0","bus1","build_year","p_nom","p_nom_min","p_nom_extendable"]].copy()
                    print(show.sort_values(["carrier","bus0","bus1"]).head(40))

            if len(inh):
                inh_key = key.loc[inh]
                dup_inh = inh_key[inh_key.duplicated(keep=False)]
                print(f"[links] inherited duplicates among themselves: {dup_inh.index.nunique()}")
                if not dup_inh.empty:
                    show = l.loc[dup_inh.index, ["carrier","bus0","bus1","build_year","p_nom","p_nom_min","p_nom_extendable"]]
                    print(show.sort_values(["carrier","bus0","bus1","build_year"]).head(60))

    else:
        print("\n=== links: skipped (missing columns carrier/bus0/bus1/build_year or empty) ===")



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
    lock_oil_electricity_assets(n) 
    continuity_merge_after_brownfield(n,year)
    audit_after_continuity_merge(n, year, carriers=("oil","ror","run-of-river","run of river","hydro ror"))

    disable_grid_expansion_if_limit_hit(n)
    elec_cfg = snakemake.config.get("electricity", {})
    apply_coal_supplier_phaseout(n, year, elec_cfg, verbose=True)
    apply_gas_trade_adjustments(n, snakemake.config)
    apply_storage_country_rules(n, snakemake.config)
    sanitize_carriers(n, snakemake.config)
    sanitize_locations(n)
    _assert_nom_bounds(n, tag=f"after coal phaseout {year}")
  
    raw = n.generators["p_nom"]
    num = pd.to_numeric(raw, errors="coerce")
    bad = num.isna()

    print("\n[debug] generators.p_nom")
    print("  bad count:", int(bad.sum()))
    print("  top raw bad values:")
    print(raw[bad].astype(str).value_counts().head(20))

    cols = [c for c in ["carrier","bus","build_year","p_nom_extendable","p_nom","p_nom_min","p_nom_max","p_nom_opt"] if c in n.generators.columns]
    print("\n  sample bad rows:")
    print(n.generators.loc[bad, cols].head(50))



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

    _fix_nan_p_nom_max(n)

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