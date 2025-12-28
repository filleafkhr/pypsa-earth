# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from itertools import dropwhile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pypsa
import pytz
import xarray as xr
from _helpers import mock_snakemake, override_component_attrs


def override_values(tech, year, dr):
    custom_res_t = pd.read_csv(
        snakemake.input["custom_res_pot_{0}_{1}_{2}".format(tech, year, dr)],
        index_col=0,
        parse_dates=True,
    ).filter(buses, axis=1)

    custom_res = (
        pd.read_csv(
            snakemake.input["custom_res_ins_{0}_{1}_{2}".format(tech, year, dr)],
            index_col=0,
        )
        .filter(buses, axis=0)
        .reset_index()
    )

    custom_res["Generator"] = custom_res["Generator"].apply(lambda x: x + " " + tech)
    custom_res = custom_res.set_index("Generator")

    if tech.replace("-", " ") in n.generators.carrier.unique():
        to_drop = n.generators[n.generators.carrier == tech].index
        n.mremove("Generator", to_drop)

    if snakemake.wildcards["planning_horizons"] == 2050:
        directory = "results/" + snakemake.params.run.replace("2050", "2030")
        n_name = snakemake.input.network.split("/")[-1].replace(
            n.config["scenario"]["clusters"], ""
        )
        df = pd.read_csv(directory + "/res_caps_" + n_name, index_col=0)
        # df = pd.read_csv(snakemake.config["custom_data"]["existing_renewables"], index_col=0)
        existing_res = df.loc[tech]
        existing_res.index = existing_res.index.str.apply(lambda x: x + tech)
    else:
        existing_res = custom_res["installedcapacity"].values

    n.madd(
        "Generator",
        buses,
        " " + tech,
        bus=buses,
        carrier=tech,
        p_nom_extendable=True,
        p_nom_max=custom_res["p_nom_max"].values,
        # weight=ds["weight"].to_pandas(),
        # marginal_cost=custom_res["fixedomEuroPKW"].values * 1000,
        capital_cost=custom_res["annualcostEuroPMW"].values,
        efficiency=1.0,
        p_max_pu=custom_res_t,
        lifetime=custom_res["lifetime"][0],
        p_nom_min=existing_res,
    )

def get_electricity_scale(snakemake) -> float:
    """
    Determine the electricity load scaling factor based on planning horizon
    defined in snakemake.wildcards.planning_horizons.

    YAML expected:
    -------------
    load_options:
      scale:
        electricity:
          2030: 1.2
          2035: 1.4
          2040: 1.6

    or a global:
      scale:
        electricity: 1.25
    """

    # 1) planning horizon is ALWAYS the year
    try:
        current_year = int(snakemake.wildcards.planning_horizons)
    except Exception:
        current_year = None
        logger.warning("⚠ could not parse planning_horizons → using global scale only")

    # 2) read config scale block
    raw_scale = (
        snakemake.params.load_options
        .get("scale", {})
        .get("electricity", 1.0)
    )

    # 3) if dictionary: match year-specific entry
    if isinstance(raw_scale, dict) and current_year in raw_scale:
        scale = float(raw_scale[current_year])
        logger.info(f"📈 electricity scale({current_year}) = {scale}")
    else:
        scale = float(raw_scale)
        if current_year:
            logger.info(f"📉 using default/global scale={scale} (no value for {current_year})")
        else:
            logger.info(f"📉 using global scale={scale} (year unknown)")

    return scale


def scale_electricity_demand(n, snakemake):
    """
    Scales electricity demand in `n.loads_t.p_set` by the factor from get_electricity_scale().

    - identifies loads with carrier == 'electricity'
      (falls back to all loads if 'carrier' is missing)
    - multiplies their time series in n.loads_t.p_set by the scale factor
    """
    scale = get_electricity_scale(snakemake)

    if np.isclose(scale, 1.0):
        logger.info("electricity demand scale ≈ 1.0, nothing to do")
        return

    if not hasattr(n, "loads_t") or "p_set" not in getattr(n.loads_t, "__dict__", {}):
        logger.warning("network has no loads_t.p_set; skipping electricity scaling")
        return

    # select electricity loads
    if "carrier" in n.loads.columns:
        mask = n.loads["carrier"].str.lower().eq("electricity")
        elec_loads = n.loads.index[mask]
        if elec_loads.empty:
            logger.warning(
                "no loads with carrier 'electricity' found; applying scale to all loads instead"
            )
            elec_loads = n.loads.index
    else:
        logger.warning(
            "loads have no 'carrier' column; applying scale to all loads"
        )
        elec_loads = n.loads.index

    # apply scaling
    before_sum = n.loads_t.p_set[elec_loads].sum().sum()
    n.loads_t.p_set[elec_loads] *= scale
    after_sum = n.loads_t.p_set[elec_loads].sum().sum()

    logger.info(
        f"scaled electricity demand for {len(elec_loads)} loads by factor {scale:.3f} "
        f"(total MWh sum: {before_sum:.2f} -> {after_sum:.2f})"
    )

if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "override_respot",
            simpl="",
            clusters="4",
            ll="c1",
            opts="Co2L-4H",
            planning_horizons="2030",
            sopts="144H",
            discountrate=0.071,
            demand="AB",
        )

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)
    m = n.copy()
    if snakemake.params.custom_data["renewables"]:
        buses = list(n.buses[n.buses.carrier == "AC"].index)
        energy_totals = pd.read_csv(snakemake.input.energy_totals, index_col=0)
        countries = snakemake.params.countries
        if snakemake.params.custom_data["renewables"]:
            techs = snakemake.params.custom_data["renewables"]
            year = snakemake.wildcards["planning_horizons"]
            dr = snakemake.wildcards["discountrate"]

            m = n.copy()

            for tech in techs:
                override_values(tech, year, dr)

        else:
            print("No RES potential techs to override...")

    if snakemake.params.load_options["scale"]["electricity"]:

        planning_year = int(snakemake.wildcards["planning_horizons"])

        year_scale_map = (
            snakemake.params.load_options["scale"]["electricity"].get(planning_year)
            or snakemake.params.load_options["scale"]["electricity"].get(str(planning_year), {})
        )

        # read energy totals (index must be country codes like BN, ID, TH, ...)
        energy_totals = pd.read_csv(snakemake.input.energy_totals, index_col=0)

        ac_buses = n.buses.index[n.buses.carrier == "AC"]
        hours = n.snapshot_weightings["objective"]  # works for 3h, 1h, etc.
        H = float(hours.sum())                      # total weighted hours

        countries = n.buses.loc[ac_buses, "country"].dropna().unique().tolist()

        for country in countries:
            # all AC loads located in this country (by their bus' country)
            loads_ct = n.loads.index[n.loads.bus.isin(ac_buses) &
                                    (n.loads.bus.map(n.buses.country) == country)]

            if len(loads_ct) == 0:
                continue

            # split: time-varying vs constant
            cols_ts = n.loads_t.p_set.columns.intersection(loads_ct)
            cols_static = loads_ct.difference(cols_ts)

            # current annual energy (MWh/a)
            current_ts_MWh = float(n.loads_t.p_set[cols_ts].mul(hours, axis=0).sum().sum()) if len(cols_ts) else 0.0
            current_static_MWh = float(n.loads.loc[cols_static, "p_set"].fillna(0.0).sum() * H) if len(cols_static) else 0.0
            current_MWh = current_ts_MWh + current_static_MWh

            if current_MWh <= 0:
                continue

            factor = float(year_scale_map.get(country, 1.0))

            # target annual electricity demand [MWh/a]
            # (use residential electricity column; fallback to relative scaling if missing)
            if country in energy_totals.index and "electricity residential" in energy_totals.columns:
                target_MWh = float(energy_totals.loc[country, "electricity residential"]) * 1e6 * factor
            else:
                target_MWh = current_MWh * factor

            scale = target_MWh / current_MWh

            # scale time series loads (preserve their within-country shape)
            if len(cols_ts):
                n.loads_t.p_set.loc[:, cols_ts] = n.loads_t.p_set[cols_ts] * scale

            # scale constant loads
            if len(cols_static):
                n.loads.loc[cols_static, "p_set"] = n.loads.loc[cols_static, "p_set"].fillna(0.0) * scale

            logger.warning(
                f"set {country} residential electricity to {target_MWh/1e6:.3f} TWh "
                f"(scale factor={factor}, applied scale={scale:.4g}) for {planning_year}"
            )
    # --- print total electricity demand of whole region (TWh/a) at the end ---

    # weighted hours (robust to 3h, 6h, etc.)
    hours = n.snapshot_weightings["objective"]
    H = float(hours.sum())

    # define "electricity demand" = all Loads on AC buses (base + any extra AC loads)
    ac_buses = n.buses.index[n.buses.carrier == "AC"]
    ac_loads = n.loads.index[n.loads.bus.isin(ac_buses)]

    # split time-varying vs constant loads
    cols_ts = n.loads_t.p_set.columns.intersection(ac_loads)
    cols_static = ac_loads.difference(cols_ts)

    # annual energy (MWh/a)
    total_ts_MWh = float(n.loads_t.p_set[cols_ts].mul(hours, axis=0).sum().sum()) if len(cols_ts) else 0.0
    total_static_MWh = float(n.loads.loc[cols_static, "p_set"].fillna(0.0).sum() * H) if len(cols_static) else 0.0
    total_MWh = total_ts_MWh + total_static_MWh

    logger.warning(f"TOTAL ELECTRICITY DEMAND (all AC loads) = {total_MWh/1e6:.3f} TWh/a")

    n.export_to_netcdf(snakemake.output[0])
