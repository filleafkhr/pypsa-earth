# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Creates the network topology from a OpenStreetMap.

Relevant Settings
-----------------

.. code:: yaml

    snapshots:

    countries:

    electricity:
        voltages:

    lines:
        types:
        s_max_pu:
        under_construction:

    links:
        p_max_pu:
        p_nom_max:
        under_construction:

    transformers:
        x:
        s_nom:
        type:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at
    :ref:`snapshots_cf`, :ref:`toplevel_cf`, :ref:`electricity_cf`, :ref:`load_options_cf`,
    :ref:`lines_cf`, :ref:`links_cf`, :ref:`transformers_cf`

Inputs
------



Outputs
-------

- ``networks/base.nc``

    .. image:: /img/base.png
        :width: 33 %

Description
-----------
"""
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


def _get_oid(df):
    if "tags" in df.columns:
        return df.tags.str.extract('"oid"=>"(\\d+)"', expand=False)
    else:
        return pd.Series(np.nan, df.index)


def get_country(df):
    if "tags" in df.columns:
        return df.tags.str.extract('"country"=>"([A-Z]{2})"', expand=False)
    else:
        return pd.Series(np.nan, df.index)


def _find_closest_links(links, new_links, distance_upper_bound=1.5):
    treecoords = np.asarray(
        [np.asarray(shapely.wkt.loads(s))[[0, -1]].flatten() for s in links.geometry]
    )
    querycoords = np.vstack(
        [new_links[["x1", "y1", "x2", "y2"]], new_links[["x2", "y2", "x1", "y1"]]]
    )
    tree = sp.spatial.KDTree(treecoords)
    dist, ind = tree.query(querycoords, distance_upper_bound=distance_upper_bound)
    found_b = ind < len(links)
    found_i = np.arange(len(new_links) * 2)[found_b] % len(new_links)

    return (
        pd.DataFrame(
            dict(D=dist[found_b], i=links.index[ind[found_b] % len(links)]),
            index=new_links.index[found_i],
        )
        .sort_values(by="D")[lambda ds: ~ds.index.duplicated(keep="first")]
        .sort_index()["i"]
    )


def _load_buses_from_osm(fp_buses):
    buses = (
        read_csv_nafix(fp_buses, dtype=dict(bus_id="str", voltage="float"))
        .set_index("bus_id")
        .drop(["station_id"], axis=1)
        .rename(columns=dict(voltage="v_nom"))
    )

    buses = buses.loc[:, ~buses.columns.str.contains("^Unnamed")]
    buses["v_nom"] /= 1e3
    buses["carrier"] = buses.pop("dc").map({True: "DC", False: "AC"})
    buses["under_construction"] = buses["under_construction"].fillna(False).astype(bool)
    buses["x"] = buses["lon"]
    buses["y"] = buses["lat"]
    # TODO: Drop NAN maybe somewhere else?
    buses = buses.dropna(axis="index", subset=["x", "y", "country"])

    return buses


def add_underwater_links(n, fp_offshore_shapes):
    if not hasattr(n.links, "geometry"):
        n.links["underwater_fraction"] = 0.0
    else:
        offshore_shape = gpd.read_file(fp_offshore_shapes).unary_union
        if offshore_shape is None or offshore_shape.is_empty:
            n.links["underwater_fraction"] = 0.0
        else:
            links = gpd.GeoSeries(n.links.geometry.dropna().map(shapely.wkt.loads))
            n.links["underwater_fraction"] = (
                links.intersection(offshore_shape).length / links.length
            )


def _set_dc_underwater_fraction(lines_or_links, fp_offshore_shapes):
    # HVDC part always has some links as converters
    # excluding probably purely DC networks which are currently somewhat exotic
    if lines_or_links.empty:
        return

    if lines_or_links.loc[lines_or_links.carrier == "DC"].empty:
        # Add "underwater_fraction" both to lines and links
        lines_or_links["underwater_fraction"] = 0.0
        return

    if not hasattr(lines_or_links, "geometry"):
        lines_or_links["underwater_fraction"] = 0.0
    else:
        offshore_shape = gpd.read_file(fp_offshore_shapes).unary_union
        if offshore_shape is None or offshore_shape.is_empty:
            lines_or_links["underwater_fraction"] = 0.0
        else:
            branches = gpd.GeoSeries(
                lines_or_links.geometry.dropna().map(shapely.wkt.loads)
            )
            # fix to avoid NaN for links during augmentation
            if branches.empty:
                lines_or_links["underwater_fraction"] = 0
            else:
                lines_or_links["underwater_fraction"] = (
                    # TODO Check assumption that all underwater lines are DC
                    branches.intersection(offshore_shape).length
                    / branches.length
                )


def _load_lines_from_osm(fp_osm_lines):
    lines = (
        read_csv_nafix(
            fp_osm_lines,
            dtype=dict(
                line_id="str",
                bus0="str",
                bus1="str",
                underground="bool",
                under_construction="bool",
                voltage="float",
                circuits="float",
            ),
        )
        .set_index("line_id")
        .rename(columns=dict(voltage="v_nom", circuits="num_parallel"))
    )

    lines["length"] /= 1e3  # m to km conversion
    lines["v_nom"] /= 1e3  # V to kV conversion
    lines = lines.loc[:, ~lines.columns.str.contains("^Unnamed")]  # remove unnamed col
    # lines = _remove_dangling_branches(lines, buses)  # TODO: add dangling branch removal?

    return lines


# TODO Seems to be not needed anymore
def _load_links_from_osm(fp_osm_converters, base_network_config, voltages_config):
    # the links file can be empty
    if os.path.getsize(fp_osm_converters) == 0:
        links = pd.DataFrame()
        return links

    links = (
        read_csv_nafix(
            fp_osm_converters,
            dtype=dict(
                line_id="str",
                bus0="str",
                bus1="str",
                underground="bool",
                under_construction="bool",
            ),
        )
        .set_index("line_id")
        .rename(columns=dict(voltage="v_nom", circuits="num_parallel"))
    )

    links["length"] /= 1e3  # m to km conversion
    links["v_nom"] /= 1e3  # V to kV conversion
    links = links.loc[:, ~links.columns.str.contains("^Unnamed")]  # remove unnamed col
    # links = _remove_dangling_branches(links, buses)  # TODO: add dangling branch removal?

    return links


def _load_converters_from_osm(fp_osm_converters, buses):
    # the links file can be empty
    if os.path.getsize(fp_osm_converters) == 0:
        converters = pd.DataFrame()
        return converters

    converters = read_csv_nafix(
        fp_osm_converters,
        dtype=dict(converter_id="str", bus0="str", bus1="str"),
    ).set_index("converter_id")

    # converters = _remove_dangling_branches(converters, buses)

    converters["carrier"] = "B2B"
    converters["dc"] = True

    return converters


def _load_transformers_from_osm(fp_osm_transformers, buses):
    transformers = (
        read_csv_nafix(
            fp_osm_transformers,
            dtype=dict(transformer_id="str", bus0="str", bus1="str"),
        )
        .rename(columns=dict(line_id="transformer_id"))
        .set_index("transformer_id")
    )
    # transformers = _remove_dangling_branches(transformers, buses)  # TODO: add dangling branch removal?

    return transformers


def _get_linetypes_config(line_types, voltages):
    """
    Return the dictionary of linetypes for selected voltages. The dictionary is
    a subset of the dictionary line_types, whose keys match the selected
    voltages.

    Parameters
    ----------
    line_types : dict
        Dictionary of linetypes: keys are nominal voltages and values are linetypes.
    voltages : list
        List of selected voltages.

    Returns
    -------
        Dictionary of linetypes for selected voltages.
    """
    # get voltages value that are not available in the line types
    vnoms_diff = set(voltages).symmetric_difference(set(line_types.keys()))
    if vnoms_diff:
        logger.warning(
            f"Voltages {vnoms_diff} not in the {line_types} or {voltages} list."
        )
    return {k: v for k, v in line_types.items() if k in voltages}


def _get_linetype_by_voltage(v_nom, d_linetypes):
    """
    Return the linetype of a specific line based on its voltage v_nom.

    Parameters
    ----------
    v_nom : float
        The voltage of the line.
    d_linetypes : dict
        Dictionary of linetypes: keys are nominal voltages and values are linetypes.

    Returns
    -------
        The linetype of the line whose nominal voltage is closest to the line voltage.
    """
    v_nom_min, line_type_min = min(
        d_linetypes.items(),
        key=lambda x: abs(x[0] - v_nom),
    )
    return line_type_min


def _set_electrical_parameters_lines(lines_config, voltages, lines):
    if lines.empty:
        lines["type"] = []
        return lines

    linetypes = _get_linetypes_config(lines_config["ac_types"], voltages)

    lines["carrier"] = "AC"
    lines["dc"] = False

    lines.loc[:, "type"] = lines.v_nom.apply(
        lambda x: _get_linetype_by_voltage(x, linetypes)
    )

    lines["s_max_pu"] = lines_config["s_max_pu"]

    return lines


def _set_electrical_parameters_dc_lines(lines_config, voltages, lines):
    if lines.empty:
        lines["type"] = []
        return lines

    linetypes = _get_linetypes_config(lines_config["dc_types"], voltages)

    lines["carrier"] = "DC"
    lines["dc"] = True
    lines.loc[:, "type"] = lines.v_nom.apply(
        lambda x: _get_linetype_by_voltage(x, linetypes)
    )

    lines["s_max_pu"] = lines_config["s_max_pu"]

    return lines


def _set_electrical_parameters_links(links_config, links):
    if links.empty:
        return links

    p_max_pu = links_config.get("p_max_pu", 1.0)
    links["p_max_pu"] = p_max_pu
    links["p_min_pu"] = -p_max_pu

    links["carrier"] = "DC"
    links["dc"] = True

    return links


def _set_electrical_parameters_transformers(transformers_config, transformers):
    config = transformers_config

    # Add transformer parameters
    transformers["x"] = config.get("x", 0.1)
    transformers["s_nom"] = config.get("s_nom", 2000)
    transformers["type"] = config.get("type", "")

    return transformers


def _set_electrical_parameters_converters(links_config, converters):
    p_max_pu = links_config.get("p_max_pu", 1.0)
    converters["p_max_pu"] = p_max_pu
    converters["p_min_pu"] = -p_max_pu

    converters["p_nom"] = 2000  # [MW]?

    # Converters are combined with links
    converters["under_construction"] = False
    converters["underground"] = False

    return converters


def _set_lines_s_nom_from_linetypes(n):
    # Info: n.line_types is a lineregister from pypsa/pandapowers
    n.lines["s_nom"] = (
        np.sqrt(3)
        * n.lines["type"].map(n.line_types.i_nom)
        * n.lines.eval("v_nom * num_parallel")
    )
    # Re-define s_nom for DC lines
    n.lines.loc[n.lines["carrier"] == "DC", "s_nom"] = n.lines["type"].map(
        n.line_types.i_nom
    ) * n.lines.eval("v_nom * num_parallel")


def _remove_dangling_branches(branches, buses):
    return pd.DataFrame(
        branches.loc[branches.bus0.isin(buses.index) & branches.bus1.isin(buses.index)]
    )


def _set_countries_and_substations(inputs, base_network_config, countries_config, n):
    countries = countries_config
    country_shapes = gpd.read_file(inputs.country_shapes).set_index("name")["geometry"]

    offshore_shapes = unary_union(gpd.read_file(inputs.offshore_shapes)["geometry"])

    buses = n.buses
    bus_locations = buses
    bus_locations = gpd.GeoDataFrame(
        bus_locations,
        geometry=gpd.points_from_xy(bus_locations.x, bus_locations.y),
        crs=country_shapes.crs,  # the workflow sets the the same crs for buses and shapes
    )
    # Check if bus is in shape
    offshore_b = bus_locations.within(offshore_shapes)

    # Assumption that HV-bus qualifies as potential offshore bus. Offshore bus is empty otherwise.
    offshore_hvb = (
        buses["v_nom"] >= base_network_config["min_voltage_substation_offshore"] / 1000
    )
    # Compares two lists & makes list value true if at least one is true
    buses["substation_off"] = offshore_b | offshore_hvb

    # Buses without country tag are removed OR get a country tag if close to country
    c_nan_b = buses.country.isnull()
    if c_nan_b.sum() > 0:
        c_tag = get_country(buses.loc[c_nan_b])
        c_tag.loc[~c_tag.isin(countries)] = np.nan
        n.buses.loc[c_nan_b, "country"] = c_tag

        c_tag_nan_b = n.buses.country.isnull()

        # Nearest country in path length defines country of still homeless buses
        # Work-around until commit 705119 lands in pypsa release
        # pypsa-earth comment: Important to connect 'homeless' offshore assets
        # Otherwise
        n.transformers["length"] = 0.0
        graph = n.graph(weight="length")
        n.transformers.drop("length", axis=1, inplace=True)

        for b in n.buses.index[c_tag_nan_b]:
            df = (
                pd.DataFrame(
                    dict(
                        pathlength=nx.single_source_dijkstra_path_length(
                            graph, b, cutoff=200
                        )
                    )
                )
                .join(n.buses.country)
                .dropna()
            )
            assert (
                not df.empty
            ), "No buses with defined country within 200km of bus `{}`".format(b)
            n.buses.at[b, "country"] = df.loc[df.pathlength.idxmin(), "country"]

        logger.warning(
            "{} buses are not in any country or offshore shape,"
            " {} have been assigned from the tag of the entsoe map,"
            " the rest from the next bus in terms of pathlength.".format(
                c_nan_b.sum(), c_nan_b.sum() - c_tag_nan_b.sum()
            )
        )

    return buses


def base_network(
    inputs,
    base_network_config,
    countries_config,
    hvdc_as_lines_config,
    lines_config,
    links_config,
    snapshots_config,
    transformers_config,
    voltages_config,
):
    buses = _load_buses_from_osm(inputs.osm_buses).reset_index(drop=True)
    lines = _load_lines_from_osm(inputs.osm_lines).reset_index(drop=True)
    transformers = _load_transformers_from_osm(inputs.osm_transformers, buses)
    converters = _load_converters_from_osm(inputs.osm_converters, buses)

    lines_ac = lines[~lines.dc].copy()
    lines_dc = lines[lines.dc].copy()
    lines_ac = _set_electrical_parameters_lines(lines_config, voltages_config, lines_ac)

    lines_dc = _set_electrical_parameters_dc_lines(
        lines_config, voltages_config, lines_dc
    )

    transformers = _set_electrical_parameters_transformers(
        transformers_config, transformers
    )
    converters = _set_electrical_parameters_converters(links_config, converters)

    n = pypsa.Network()
    n.name = "PyPSA-Earth"

    n.set_snapshots(pd.date_range(freq="h", **snapshots_config))
    n.snapshot_weightings[:] *= 8760.0 / n.snapshot_weightings.sum()

    n.import_components_from_dataframe(buses, "Bus")

    if hvdc_as_lines_config:
        lines = pd.concat([lines_ac, lines_dc])
        n.import_components_from_dataframe(lines, "Line")
    else:
        lines_dc = _set_electrical_parameters_links(links_config, lines_dc)
        # parse line information into p_nom required for converters
        lines_dc["p_nom"] = lines_dc.apply(
            lambda x: x["v_nom"] * n.line_types.i_nom[x["type"]],
            axis=1,
            result_type="reduce",
        )
        n.import_components_from_dataframe(lines_ac, "Line")
        n.import_components_from_dataframe(lines_dc, "Link")

    n.import_components_from_dataframe(transformers, "Transformer")
    n.import_components_from_dataframe(converters, "Link")

    _set_lines_s_nom_from_linetypes(n)

    _set_countries_and_substations(inputs, base_network_config, countries_config, n)

    _set_dc_underwater_fraction(n.lines, inputs.offshore_shapes)
    _set_dc_underwater_fraction(n.links, inputs.offshore_shapes)
# ---- CLEANUP: cross-border normalization WITHOUT changing bus0/bus1 ----
    if base_network_config.get("cleanup", False):

        raw_whitelist = base_network_config.get("cleanup_whitelist") or []
        whitelist_pairs = {tuple(sorted(p)) for p in raw_whitelist}

        def compute_crossborder_scale_factors(
            n,
            cap_selector: str = "max",            # {"max","p95"}
            s_max_pu_default: float = 1.0,
            whitelist_pairs: set | None = None,   # set of tuples like {("DE","DK"), ("MY","ID")}
        ):
            """
            Return dict with two pd.Series of per-asset scale factors (index = asset index):
            {"lines": scale_series_for_lines, "links": scale_series_for_links}

            Factors are computed so that, for each (carrier, country0, country1) corridor,
            the SUM of capacities after scaling equals the selected corridor target capacity.
            Whitelisted country pairs get factor = 1.0 (no change).
            """

            def _recompute_country_cols(n, tbl: str, canonize_for_clustering: bool = True):
                df = getattr(n, tbl)
                if df.empty:
                    return df
                b_country = n.buses["country"]
                df = df.copy()
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
                return df

            def _select_cap(arr: pd.Series, mode: str) -> float:
                vals = pd.to_numeric(arr, errors="coerce").dropna().values
                if vals.size == 0:
                    return np.nan
                return float(np.percentile(vals, 95)) if mode == "p95" else float(np.max(vals))

            def _scale_series_for_table(table: str, cap_col: str) -> pd.Series:
                df = getattr(n, table)
                if df.empty:
                    return pd.Series(dtype=float)

                df = _recompute_country_cols(n, table, True)

                if cap_col not in df.columns:
                    logger.info(f"{table}: '{cap_col}' not found; returning neutral factors.")
                    return pd.Series(1.0, index=df.index, dtype=float)

                # numeric caps
                caps = pd.to_numeric(df[cap_col], errors="coerce").fillna(0.0)
                df = df.copy()
                df[cap_col] = caps

                # only cross-border
                cross = df["country0"].notna() & df["country1"].notna() & (df["country0"] != df["country1"])
                if not cross.any():
                    logger.info(f"{table}: no cross-border assets; returning neutral factors.")
                    return pd.Series(1.0, index=df.index, dtype=float)

                work = df.loc[cross].copy()

                # carrier fallback
                if "carrier" not in work.columns:
                    work["carrier"] = "AC" if table == "lines" else "DC"

                # optional whitelist
                if whitelist_pairs:
                    work["skip"] = [(a, b) in whitelist_pairs for a, b in zip(work["country0"], work["country1"])]
                else:
                    work["skip"] = False

                grp_cols = ["carrier", "country0", "country1"]

                # selected corridor cap (max or p95) per corridor
                work["corr_cap_sel"] = work.groupby(grp_cols)[cap_col].transform(lambda s: _select_cap(s, cap_selector))

                # physics cap for AC lines (optional clamp)
                if table == "lines":
                    # ensure s_max_pu
                    if "s_max_pu" not in work.columns:
                        work["s_max_pu"] = s_max_pu_default
                    # need v_nom & i_nom — if missing, physics cap stays NaN (ignored)
                    if ("v_nom" in work.columns) and ("i_nom" in work.columns):
                        m = work["v_nom"].notna() & work["i_nom"].notna()
                        work.loc[m, "s_cap_seg"] = np.sqrt(3.0) * work.loc[m, "v_nom"] * work.loc[m, "i_nom"] * work.loc[m, "s_max_pu"]
                        work["phys_cap_corr"] = work.groupby(grp_cols)["s_cap_seg"].transform("max")
                        # apply physics clamp where available
                        use_phys = work["phys_cap_corr"].notna()
                        work.loc[use_phys, "corr_cap_sel"] = np.minimum(
                            work.loc[use_phys, "corr_cap_sel"], work.loc[use_phys, "phys_cap_corr"]
                        )

                # corridor sum of current capacities (denominator)
                work["sum_cap_corr"] = work.groupby(grp_cols)[cap_col].transform("sum").replace(0.0, np.nan)

                # corridor factor: target / current
                work["corr_factor"] = work["corr_cap_sel"] / work["sum_cap_corr"]

                # rows to skip (whitelist) → factor 1.0
                work.loc[work["skip"], "corr_factor"] = 1.0

                # if denom was NaN (all zeros), keep neutral factor
                work["corr_factor"] = work["corr_factor"].fillna(1.0)

                # assemble full series (1.0 outside cross-border)
                factors = pd.Series(1.0, index=df.index, dtype=float)
                factors.loc[work.index] = work["corr_factor"].astype(float).values
                return factors

            factors_lines = _scale_series_for_table("lines", "s_nom")
            factors_links = _scale_series_for_table("links", "p_nom")

            return {"lines": factors_lines, "links": factors_links}


        def apply_crossborder_scales(n, factors: dict):
            """
            Multiply capacities by the provided per-asset factors.
            factors["lines"]: pd.Series indexed by n.lines.index (scales s_nom)
            factors["links"]: pd.Series indexed by n.links.index (scales p_nom)
            """
            # lines
            if not n.lines.empty and "lines" in factors and not factors["lines"].empty:
                scale = factors["lines"].reindex(n.lines.index).fillna(1.0).astype(float)
                before = pd.to_numeric(n.lines["s_nom"], errors="coerce").fillna(0.0).copy()
                n.lines["s_nom"] = (before * scale).astype(float)

            # links
            if not n.links.empty and "links" in factors and not factors["links"].empty:
                scale = factors["links"].reindex(n.links.index).fillna(1.0).astype(float)
                before = pd.to_numeric(n.links["p_nom"], errors="coerce").fillna(0.0).copy()
                n.links["p_nom"] = (before * scale).astype(float)

        # 1) compute scale factors (does NOT modify the network)
        factors = compute_crossborder_scale_factors(
            n,
            cap_selector="max",                    # or "p95"
            s_max_pu_default=1.0,
            whitelist_pairs=whitelist_pairs         # optional
        )

        # 2) apply (multiplicative) scaling to lines/links
        apply_crossborder_scales(n, factors)

    return n

def drop_interconnectors(n):
    """
    Remove all lines and links that connect buses in different countries.
    """
    # Ensure bus countries are available
    bus_country = n.buses["country"]

    # For lines
    if not n.lines.empty:
        c0 = bus_country.reindex(n.lines["bus0"]).values
        c1 = bus_country.reindex(n.lines["bus1"]).values
        mask = (c0 == c1)
        n.lines = n.lines[mask]

    # For links
    if not n.links.empty:
        c0 = bus_country.reindex(n.links["bus0"]).values
        c1 = bus_country.reindex(n.links["bus1"]).values
        mask = (c0 == c1)
        n.links = n.links[mask]
    
    print('cleaned interconnections')

    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("base_network")

    configure_logging(snakemake)

    inputs = snakemake.input

    # Snakemake imports:
    base_network_config = snakemake.params.base_network
    countries = snakemake.params.countries
    hvdc_as_lines = snakemake.params.hvdc_as_lines
    lines = snakemake.params.lines
    links = snakemake.params.links
    snapshots = snakemake.params.snapshots
    transformers = snakemake.params.transformers
    voltages = snakemake.params.voltages

    n = base_network(
        inputs,
        base_network_config,
        countries,
        hvdc_as_lines,
        lines,
        links,
        snapshots,
        transformers,
        voltages,
    )

    n.buses = pd.DataFrame(n.buses.drop(columns="geometry"))
    n.meta = snakemake.config
    if base_network_config['clean_interconnectors'] == True:
        n = drop_interconnectors(n)
    n.export_to_netcdf(snakemake.output[0])
