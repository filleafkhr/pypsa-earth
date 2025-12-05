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
from add_existing_baseyear import add_build_year_to_new_assets

# from pypsa.clustering.spatial import normed_or_uniform

logger = logging.getLogger(__name__)
idx = pd.IndexSlice
def add_curtailment_penalty(n, curtailment_cost_eur_per_mwh=100.0, sink_bus="curtailment_sink"):
    # global sink bus
    if sink_bus not in n.buses.index:
        n.add("Bus", sink_bus, carrier="curtailment")

    # per-AC bus link to sink with a positive marginal cost (penalty)
    ac_buses = n.buses.index[n.buses.carrier == "AC"]
    link_names = [f"{b} -> {sink_bus} curtailment" for b in ac_buses]

    n.madd(
        "Link",
        link_names,
        bus0=ac_buses,
        bus1=sink_bus,
        carrier="curtailment",
        p_nom_extendable=True,
        p_min_pu=0.0,
        efficiency=0.0,                 # energy disappears
        marginal_cost=curtailment_cost_eur_per_mwh
    )


def apply_coal_supplier_phaseout(n, year, elec_cfg, per_bus_factors=None, verbose=True):
    """
    One-shot coal/lignite supplier-side phaseout:
      • reads elec_cfg['coal_phaseout'] (enable, start_year, target_year{year: frac})
      • computes the fraction for `year` via piecewise-linear interpolation
      • rescales supplier-side capacity:
          - Generators: all with name/carrier matching coal|lignite
          - Links: those with name/carrier matching coal|lignite OR with bus0 on a coal/lignite bus
        For extendable assets: scales p_nom_max (and keeps p_nom <= p_nom_max).
        For fixed assets: scales p_nom.
      • Optional `per_bus_factors` (dict bus->factor) overrides the global fraction per bus0/bus.

    Parameters
    ----------
    n : pypsa.Network
    year : int
    elec_cfg : dict
      e.g.
        coal_phaseout:
          enable: true
          start_year: 2030
          target_year:
            2040: 0.5
            2050: 0.0
          min_age: 10   # ignored here
    per_bus_factors : dict[str,float] | None
    verbose : bool

    Returns
    -------
    float
        applied global fraction for this year
    """
    # ---- helpers (no extra imports) ----
    def _cfg_fraction(y:int, cfg:dict) -> float:
        cp = (cfg or {}).get("coal_phaseout", {}) or {}
        if not cp.get("enable", False):
            return 1.0
        try:
            start = int(cp.get("start_year", y))
        except Exception:
            start = y

        # targets like {"2040":0.5,"2050":0.0} -> {2040:0.5, 2050:0.0}
        raw = cp.get("target_year") or {}
        targets = {int(k): float(v) for k, v in raw.items()}
        if not targets:
            return 1.0

        if y <= start:
            return 1.0
        items = sorted(targets.items())  # [(2040,0.5),(2050,0.0)]
        if y >= items[-1][0]:
            return items[-1][1]

        y1, f1 = start, 1.0
        for y2, f2 in items:
            if y <= y2:
                t = (y - y1) / float(y2 - y1)
                return f1 + t*(f2 - f1)
            y1, f1 = y2, f2
        return f1

    def _icontains(series, patt):
        s = series if isinstance(series, pd.Series) else pd.Series(series, index=n.links.index if hasattr(n, "links") else None)
        return s.astype(str).str.contains(patt, case=False, regex=True, na=False)

    def _coal_like_df(df):
        name = df["name"] if "name" in df else pd.Series("", index=df.index)
        carr = df["carrier"] if "carrier" in df else pd.Series("", index=df.index)
        patt = r"\b(?:coal|lignite)\b"
        return name.astype(str).str.contains(patt, case=False, regex=True, na=False) | \
               carr.astype(str).str.contains(patt, case=False, regex=True, na=False)

    def _scale_cap(df, mask, factor_like):
        if getattr(mask, "any", lambda: False)():
            # factor_like can be scalar or Series aligned to df.index
            if np.isscalar(factor_like):
                fac = pd.Series(factor_like, index=df.index)
            else:
                fac = pd.Series(factor_like).reindex(df.index).fillna(1.0)

            ext = df.get("p_nom_extendable", False)
            ext = ext.fillna(False) if isinstance(ext, pd.Series) else pd.Series(False, index=df.index)

            ext_mask = mask & ext
            fix_mask = mask & (~ext)

            if ext_mask.any():
                if "p_nom_max" in df:
                    df.loc[ext_mask, "p_nom_max"] *= fac.loc[ext_mask]
                if "p_nom_min" in df:
                    df.loc[ext_mask, "p_nom_min"] *= fac.loc[ext_mask]
                if "p_nom" in df and "p_nom_max" in df:
                    df.loc[ext_mask, "p_nom"] = np.minimum(df.loc[ext_mask, "p_nom"], df.loc[ext_mask, "p_nom_max"])

            if fix_mask.any() and "p_nom" in df:
                df.loc[fix_mask, "p_nom"] *= fac.loc[fix_mask]

    # ---- compute fraction from config ----
    fraction = _cfg_fraction(int(year), elec_cfg)

    # ---- identify coal buses (supplier side for links) ----
    coal_bus_mask = n.buses["carrier"].astype(str).str.contains(r"\b(?:coal|lignite)\b", case=False, regex=True, na=False) \
                    if ("carrier" in n.buses) else pd.Series(False, index=n.buses.index)
    coal_buses = n.buses.index[coal_bus_mask]

    # optional per-bus overrides
    per_bus_factors = per_bus_factors or {}

    # ---- GENERATORS (suppliers by definition) ----
    if not n.generators.empty:
        g_mask = _coal_like_df(n.generators)
        if g_mask.any():
            if per_bus_factors:
                g_fac = pd.Series([per_bus_factors.get(b, fraction) for b in n.generators["bus"]], index=n.generators.index)
                _scale_cap(n.generators, g_mask, g_fac)
            else:
                _scale_cap(n.generators, g_mask, fraction)

    # ---- LINKS (scale only those that *consume* coal on input/bus0) ----
    if not n.links.empty:
        # coal-like by name/carrier OR bus0 in a coal bus
        l_mask_name = _coal_like_df(n.links)
        l_mask_bus0 = n.links["bus0"].isin(coal_buses) if "bus0" in n.links else pd.Series(False, index=n.links.index)
        l_mask = l_mask_name | l_mask_bus0
        if l_mask.any():
            if per_bus_factors and "bus0" in n.links:
                l_fac = pd.Series([per_bus_factors.get(b0, fraction) for b0 in n.links["bus0"]], index=n.links.index)
                _scale_cap(n.links, l_mask, l_fac)
            else:
                _scale_cap(n.links, l_mask, fraction)

    if verbose:
        def _sum(df, sel):
            return float(df.loc[sel, "p_nom"].sum()) if ("p_nom" in df and sel.any()) else 0.0
        g_sel = _coal_like_df(n.generators) if not n.generators.empty else pd.Series([], dtype=bool)
        l_sel = _coal_like_df(n.links) if not n.links.empty else pd.Series([], dtype=bool)
        g_cap = _sum(n.generators, g_sel)
        l_cap = _sum(n.links, l_sel)
        print(f"[coal phaseout] year={year} factor={fraction:.3f} | "
              f"[supplier-side] gen≈{g_cap:.3e} MW, link≈{l_cap:.3e} MW")

    return fraction


def add_brownfield(n, n_p, year):
    logger.info(f"Preparing brownfield for the year {year}")

    # electric transmission grid set optimised capacities of previous as minimum
    n.lines.s_nom_min = n_p.lines.s_nom_opt
    dc_i = n.links[n.links.carrier == "DC"].index
    n.links.loc[dc_i, "p_nom_min"] = n_p.links.loc[dc_i, "p_nom_opt"]

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
    elec_cfg = snakemake.config.get("electricity", {})
    apply_coal_supplier_phaseout(n, year, elec_cfg, verbose=True)
    # === REBALANCE MIX (no external 'costs' needed) ===


    def add_curtailment_penalty(n, penalty=100.0, sink_bus="curtailment_sink"):
        if sink_bus not in n.buses.index:
            n.add("Bus", sink_bus, carrier="curtailment")
        ac_buses = n.buses.index[n.buses.carrier == "AC"]
        link_names = [f"{b} -> {sink_bus} curtailment" for b in ac_buses]
        to_add = [ln for ln in link_names if ln not in n.links.index]
        if to_add:
            sel = [ln.split(" -> ")[0] for ln in to_add]
            n.madd("Link", to_add,
                bus0=sel, bus1=sink_bus,
                carrier="curtailment",
                p_nom_extendable=True, p_min_pu=0.0,
                efficiency=0.0, marginal_cost=penalty)
        else:
            n.links.loc[n.links.carrier=="curtailment","marginal_cost"] = penalty

    def guess_biomass_intensity(n, default=0.35):
        """tCO2 per MWh_fuel. Uses carriers.co2_emissions if present; else default."""
        try:
            if hasattr(n, "carriers") and "co2_emissions" in n.carriers.columns:
                for label in ["solid biomass","biomass","wood"]:
                    if label in n.carriers.index:
                        val = n.carriers.at[label, "co2_emissions"]
                        if pd.notna(val):
                            return float(val)
        except Exception:
            pass
        return float(default)

    def rebalance_generation(
        n,
        curtailment_cost=100.0,
        solar_mc=3.0,
        h2_el_mc=2.0,
        h2_node_cap_mw=500,
        h2_rt_eff=0.95,
        h2_store_capex_mult=1.5,
        onwind_cap_mw=3000,
        offac_cap_mw=5000,
        offdc_cap_mw=5000,
        solar_site_cap_mw=3000,
        beop_cap_mw=300,
        co2_store_capex_mult=2.0,
        co2_store_e_nom_max_t=5e8,
        biomass_intensity_t_per_mwh=None,
    ):
        # 1) penalise curtailment
        add_curtailment_penalty(n, curtailment_cost)

        # 2) H2 electrolysis not bottomless
        m_el = n.links.carrier.str.contains("H2 Electrolysis", case=False, regex=False)
        if m_el.any():
            n.links.loc[m_el, "marginal_cost"] = h2_el_mc
            n.links.loc[m_el, "p_nom_max"] = (
                n.links.loc[m_el, "p_nom_max"].replace([np.inf], np.nan).fillna(h2_node_cap_mw).clip(upper=h2_node_cap_mw)
            )
            # (optional) tie per-node cap to 30% of node peak load if higher
            if not n.loads_t.p_set.empty:
                node_peak = n.loads_t.p_set.groupby(n.loads.bus, axis=1).sum().max()
                for bus, peak in node_peak.items():
                    idx = n.links.index[m_el & (n.links.bus1 == f"{bus} H2")]
                    if len(idx):
                        n.links.loc[idx, "p_nom_max"] = np.maximum(
                            n.links.loc[idx, "p_nom_max"].fillna(0).values,
                            0.3 * float(peak)
                        )

        # H2 storage cost & round-trip
        for carr in ["H2 UHS charger","H2 UHS discharger"]:
            m = n.links.carrier.str.contains(carr, case=False, regex=False)
            if m.any():
                n.links.loc[m, "efficiency"] = h2_rt_eff
        m_h2_store = n.stores.carrier.isin(["H2 UHS","H2 Store Tank"])
        if m_h2_store.any():
            n.stores.loc[m_h2_store, "capital_cost"] = n.stores.loc[m_h2_store, "capital_cost"] * h2_store_capex_mult

        # 3) unlock wind with finite per-site caps
        for tech, cap in [("onwind", onwind_cap_mw), ("offwind-ac", offac_cap_mw), ("offwind-dc", offdc_cap_mw)]:
            m = n.generators.carrier.str.contains(tech, case=False, regex=False)
            if m.any():
                n.generators.loc[m, "p_nom_extendable"] = True
                pmax = n.generators.loc[m, "p_nom_max"].replace([np.inf], np.nan).fillna(cap)
                n.generators.loc[m, "p_nom_max"] = pmax

        # 4) solar cap + small marginal cost
        m_sol = n.generators.carrier.str.contains("solar", case=False, regex=False)
        if m_sol.any():
            pmax = n.generators.loc[m_sol, "p_nom_max"].replace([np.inf], np.nan).fillna(solar_site_cap_mw)
            n.generators.loc[m_sol, "p_nom_max"] = pmax
            n.generators.loc[m_sol, "marginal_cost"] = solar_mc

        # 5) biomass EOP emits CO2; cap biomass; bound CO2 store
        if biomass_intensity_t_per_mwh is None:
            biomass_intensity_t_per_mwh = guess_biomass_intensity(n, default=0.35)

        # ensure columns exist
        for col in ["bus2","efficiency2"]:
            if col not in n.links.columns:
                n.links[col] = np.nan

        m_beop = n.links.carrier.str.contains("biomass EOP", case=False, regex=False)
        if m_beop.any():
            n.links.loc[m_beop, "bus2"] = "co2 atmosphere"
            n.links.loc[m_beop, "efficiency2"] = biomass_intensity_t_per_mwh
            cur = n.links.loc[m_beop, "p_nom_max"].replace([np.inf], np.nan).fillna(0)
            n.links.loc[m_beop, "p_nom_max"] = np.where(cur <= 0, beop_cap_mw, cur)

        m_co2s = n.stores.carrier == "co2 stored"
        if m_co2s.any():
            n.stores.loc[m_co2s, "capital_cost"] = n.stores.loc[m_co2s, "capital_cost"] * co2_store_capex_mult
            n.stores.loc[m_co2s, "e_nom_extendable"] = True
            n.stores.loc[m_co2s, "e_nom_max"] = co2_store_e_nom_max_t

        # 6) allow grid expansion so wind isn’t stranded
        if "s_nom_extendable" in n.lines:
            n.lines["s_nom_extendable"] = True
        m_dc = n.links.carrier.str.contains("DC", case=False, regex=False)
        if m_dc.any():
            n.links.loc[m_dc, "p_nom_extendable"] = True

        # 7) helpful warnings
        if hasattr(n.generators_t, "p_max_pu"):
            for tech in ["onwind","offwind"]:
                gens = n.generators.index[n.generators.carrier.str.contains(tech, case=False, regex=False)]
                missing = [g for g in gens if g not in n.generators_t.p_max_pu.columns]
                if missing:
                    print(f"[warn] missing {tech} CF for {len(missing)} sites → they won't build")

    # run the adjustments
    rebalance_generation(n)

    # quick sanity prints (pre-solve)
    try:
        print("curtailment penalty:", float(n.links.loc[n.links.carrier=="curtailment","marginal_cost"].iloc[0]), "€/MWh")
    except Exception:
        pass
    print("H2 electrolysis sites:", int((n.links.carrier.str.contains('H2 Electrolysis', case=False, regex=False)).sum()))
    print("solar sites capped:", int((n.generators.carrier.str.contains('solar', case=False, regex=False)).sum()))
    # === module: apply_cost_overrides ===

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

        # component tables and their country series (indices match each df)
        G, L, S = n.generators, n.links, n.stores
        g_country = _country_of(n, G, "Generator") if not G.empty else pd.Series([], index=G.index, dtype=str)
        l_country = _country_of(n, L, "Link")      if not L.empty else pd.Series([], index=L.index, dtype=str)
        s_country = _country_of(n, S, "Store")     if not S.empty else pd.Series([], index=S.index, dtype=str)

        def build_mask(df, country_series, country_filter, carriers):
            """Return a boolean mask over df.index for this rule."""
            if df.empty:
                return pd.Series(False, index=df.index)

            # carrier mask
            if carriers == ["*"]:
                mask_carrier = pd.Series(True, index=df.index)
            else:
                carriers_lower = {c.lower() for c in carriers}
                mask_carrier = df["carrier"].str.lower().isin(carriers_lower)

            # country mask
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
            """Apply one YAML rule to one component df."""
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

            # enable / disable (extendability + caps)
            if enable is not None:
                enable_bool = bool(enable)

                # power capacity (Links / some Generators)
                if "p_nom_extendable" in df.columns:
                    df.loc[m, "p_nom_extendable"] = enable_bool
                    if "p_nom_max" in df.columns:
                        if enable_bool:
                            df.loc[m, "p_nom_max"] = np.inf
                        elif "p_nom" in df.columns:
                            df.loc[m, "p_nom_max"] = df.loc[m, "p_nom"]

                # energy capacity (Stores)
                if "e_nom_extendable" in df.columns:
                    df.loc[m, "e_nom_extendable"] = enable_bool
                    if "e_nom_max" in df.columns:
                        if enable_bool:
                            df.loc[m, "e_nom_max"] = np.inf
                        elif "e_nom" in df.columns:
                            df.loc[m, "e_nom_max"] = df.loc[m, "e_nom"]

            # multipliers (e.g. 0.95 * capital_cost)
            for k, fac in mult.items():
                if k in df.columns:
                    df.loc[m, k] = df.loc[m, k] * float(fac)

            # absolute overrides
            for k, val in fields.items():
                if k in df.columns:
                    df.loc[m, k] = val

        # apply every rule to every component
        for r in rules:
            apply_rule_to_df(G, g_country, r)
            apply_rule_to_df(L, l_country, r)
            apply_rule_to_df(S, s_country, r)

        print(f"✅ applied {len(rules)} storage_country_rules")

    def apply_gas_trade_adjustments(n, cfg):
        """
        Adjust gas-related costs based on importer/exporter role.

        YAML example:
        -------------
        feature_switches:
        gas_trade_adjustments: true

        gas_trade:
        importers: ["ID","PH","VN"]
        exporters: ["MY","BN"]
        importer_adjustment:
            fuel_marginal_cost_add: 3.0   # €/MWh_fuel to add to gas fuel generators
            ccgt_vom_add: 0.3             # €/MWh_e to add on CCGT link marginal_cost
            ocgt_vom_add: 0.3
            ccgt_capex_mult: 1.05
            ocgt_capex_mult: 1.05
        exporter_adjustment:
            fuel_marginal_cost_add: -2.0
            ccgt_vom_add: -0.2
            ocgt_vom_add: -0.2
            ccgt_capex_mult: 0.97
            ocgt_capex_mult: 0.97
        """
        if not cfg.get("feature_switches", {}).get("gas_trade_adjustments", False):
            print("⚙️  gas_trade_adjustments: OFF")
            return

        trade = cfg.get("gas_trade", {})
        importers = list(trade.get("importers", []))
        exporters = list(trade.get("exporters", []))
        imp_adj = trade.get("importer_adjustment", {}) or {}
        exp_adj = trade.get("exporter_adjustment", {}) or {}

        # -------- 1) fuel price: n.generators (gas fuel) --------
        g = n.generators
        if not g.empty and "carrier" in g.columns:
            g_country = _country_of(n, g, "FuelGenerator")
            is_gas_fuel = g.carrier.str.lower().eq("gas")

            if importers and "fuel_marginal_cost_add" in imp_adj:
                m = is_gas_fuel & g_country.isin(importers)
                if m.any():
                    g.loc[m, "marginal_cost"] += float(imp_adj["fuel_marginal_cost_add"])

            if exporters and "fuel_marginal_cost_add" in exp_adj:
                m = is_gas_fuel & g_country.isin(exporters)
                if m.any():
                    g.loc[m, "marginal_cost"] += float(exp_adj["fuel_marginal_cost_add"])

        # -------- 2) plant economics: n.links (CCGT / OCGT) --------
        L = n.links
        if L.empty:
            return

        l_country = _country_of(n, L, "Link")
        is_ccgt = L.carrier.str.lower().eq("ccgt")
        is_ocgt = L.carrier.str.lower().eq("ocgt")

        def _apply_link_adjust(countries, adj):
            if not countries or not adj:
                return
            mask_country = l_country.isin(list(countries))

            # VOM (marginal_cost) adders
            if "ccgt_vom_add" in adj:
                m = mask_country & is_ccgt
                if m.any():
                    L.loc[m, "marginal_cost"] += float(adj["ccgt_vom_add"])
            if "ocgt_vom_add" in adj:
                m = mask_country & is_ocgt
                if m.any():
                    L.loc[m, "marginal_cost"] += float(adj["ocgt_vom_add"])

            # CAPEX multipliers
            if "ccgt_capex_mult" in adj:
                m = mask_country & is_ccgt
                if m.any():
                    L.loc[m, "capital_cost"] *= float(adj["ccgt_capex_mult"])
            if "ocgt_capex_mult" in adj:
                m = mask_country & is_ocgt
                if m.any():
                    L.loc[m, "capital_cost"] *= float(adj["ocgt_capex_mult"])

        _apply_link_adjust(importers, imp_adj)
        _apply_link_adjust(exporters, exp_adj)

        print("✅ applied gas_trade_adjustments")


    apply_gas_trade_adjustments(n, snakemake.config)
    apply_storage_country_rules(n, snakemake.config)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
