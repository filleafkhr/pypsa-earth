# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
import logging
import os
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pypsa
import pytz
import ruamel.yaml
import xarray as xr
from _helpers import (
    BASE_DIR,
    create_dummy_data,
    create_network_topology,
    cycling_shift,
    locate_bus,
    mock_snakemake,
    override_component_attrs,
    prepare_costs,
    safe_divide,
    sanitize_carriers,
    sanitize_locations,
    three_2_two_digits_country,
    two_2_three_digits_country,
)
from prepare_transport_data import prepare_transport_data

logger = logging.getLogger(__name__)

spatial = SimpleNamespace()


def add_lifetime_wind_solar(n, costs):
    """
    Add lifetime for solar and wind generators.
    """
    for carrier in ["solar", "onwind", "offwind"]:
        gen_i = n.generators.index.str.contains(carrier)
        n.generators.loc[gen_i, "lifetime"] = costs.at[carrier, "lifetime"]


def add_carrier_buses(n, carrier, nodes=None):
    """
    Add buses to connect e.g. coal, nuclear and oil plants.
    """

    if nodes is None:
        nodes = vars(spatial)[carrier].nodes
    location = vars(spatial)[carrier].locations

    # skip if carrier already exists
    if carrier in n.carriers.index:
        return

    if not isinstance(nodes, pd.Index):
        nodes = pd.Index(nodes)

    n.add("Carrier", carrier, co2_emissions=costs.at[carrier, "CO2 intensity"])

    n.madd("Bus", nodes, location=location, carrier=carrier)


    n.madd(
        "Generator",
        nodes,
        bus=nodes,
        p_nom_extendable=True,
        carrier=carrier,
        marginal_cost=costs.at[carrier, "fuel"],
    )


def add_generation(
    n, costs, existing_capacities=0, existing_efficiencies=None, existing_nodes=None
):
    """
    Adds conventional generation as specified in config.

    Args:
        n (network): PyPSA prenetwork
        costs (dataframe): _description_
        existing_capacities: dictionary containing installed capacities for conventional_generation technologies
        existing_efficiencies: dictionary containing efficiencies for conventional_generation technologies
        existing_nodes: dictionary containing nodes for conventional_generation technologies

    Returns:
        _type_: _description_
    """ """"""

    logger.info("adding electricity generation")

    # Not required, because nodes are already defined in "nodes"
    # nodes = pop_layout.index

    fallback = {"OCGT": "gas"}
    conventionals = options.get("conventional_generation", fallback)

    for generator, carrier in conventionals.items():
        add_carrier_buses(n, carrier)
        carrier_nodes = vars(spatial)[carrier].nodes
        link_names = spatial.nodes + " " + generator
        n.madd(
            "Link",
            link_names,
            bus0=carrier_nodes,
            bus1=spatial.nodes,
            bus2="co2 atmosphere",
            marginal_cost=costs.at[generator, "efficiency"]
            * costs.at[generator, "VOM"],  # NB: VOM is per MWel
            # NB: fixed cost is per MWel
            capital_cost=costs.at[generator, "efficiency"]
            * costs.at[generator, "fixed"],
            p_nom_extendable=(
                True
                if generator
                in snakemake.params.electricity.get("extendable_carriers", dict()).get(
                    "Generator", list()
                )
                else False
            ),
            p_nom=(
                (
                    existing_capacities[generator] / existing_efficiencies[generator]
                ).reindex(link_names, fill_value=0)
                if not existing_capacities == 0
                else 0
            ),  # NB: existing capacities are MWel
            carrier=generator,
            efficiency=(
                existing_efficiencies[generator].reindex(
                    link_names, fill_value=costs.at[generator, "efficiency"]
                )
                if existing_efficiencies is not None
                else costs.at[generator, "efficiency"]
            ),
            efficiency2=costs.at[carrier, "CO2 intensity"],
            lifetime=costs.at[generator, "lifetime"],
        )

        # set the "co2_emissions" of the carrier to 0, as emissions are accounted by link efficiency separately (efficiency to 'co2 atmosphere' bus)
        n.carriers.loc[carrier, "co2_emissions"] = 0


def H2_liquid_fossil_conversions(n, costs):
    """
    Function to add conversions between H2 and liquid fossil Carrier and bus is
    added in add_oil, which later on might be switched to add_generation.
    """

    n.madd(
        "Link",
        spatial.nodes + " Fischer-Tropsch",
        bus0=spatial.nodes + " H2",
        bus1=spatial.oil.nodes,
        bus2=spatial.co2.nodes,
        bus3=spatial.nodes,
        carrier="Fischer-Tropsch",
        efficiency=costs.at["Fischer-Tropsch", "efficiency"],
        capital_cost=costs.at["Fischer-Tropsch", "fixed"]
        * costs.at[
            "Fischer-Tropsch", "efficiency"
        ],  # Use efficiency to convert from EUR/MW_FT/a to EUR/MW_H2/a
        efficiency2=-costs.at["oil", "CO2 intensity"]
        * costs.at["Fischer-Tropsch", "efficiency"],
        efficiency3=-costs.at["Fischer-Tropsch", "electricity-input"]
        / costs.at["Fischer-Tropsch", "hydrogen-input"],
        p_nom_extendable=True,
        p_min_pu=options.get("min_part_load_fischer_tropsch", 0),
        lifetime=costs.at["Fischer-Tropsch", "lifetime"],
    )


def add_hydrogen(n, costs):
    "function to add hydrogen as an energy carrier with its conversion technologies from and to AC"
    logger.info("Adding hydrogen")

    n.add("Carrier", "H2")

    n.madd(
        "Bus",
        spatial.nodes + " H2",
        location=spatial.nodes,
        carrier="H2",
        x=n.buses.loc[list(spatial.nodes)].x.values,
        y=n.buses.loc[list(spatial.nodes)].y.values,
    )

    if snakemake.config["sector"]["hydrogen"]["hydrogen_colors"]:
        n.madd(
            "Bus",
            nodes + " grid H2",
            location=nodes,
            carrier="grid H2",
            x=n.buses.loc[list(nodes)].x.values,
            y=n.buses.loc[list(nodes)].y.values,
        )

        n.madd(
            "Link",
            nodes + " H2 Electrolysis",
            bus0=nodes,
            bus1=nodes + " grid H2",
            p_nom_extendable=True,
            carrier="H2 Electrolysis",
            efficiency=costs.at["electrolysis", "efficiency"],
            capital_cost=costs.at["electrolysis", "fixed"],
            lifetime=costs.at["electrolysis", "lifetime"],
        )

        n.madd(
            "Link",
            nodes + " grid H2",
            bus0=nodes + " grid H2",
            bus1=nodes + " H2",
            p_nom_extendable=True,
            carrier="grid H2",
            efficiency=1,
            capital_cost=0,
        )

    else:
        n.madd(
            "Link",
            nodes + " H2 Electrolysis",
            bus1=nodes + " H2",
            bus0=nodes,
            p_nom_extendable=True,
            carrier="H2 Electrolysis",
            efficiency=costs.at["electrolysis", "efficiency"],
            capital_cost=costs.at["electrolysis", "fixed"],
            lifetime=costs.at["electrolysis", "lifetime"],
        )

    n.madd(
        "Link",
        spatial.nodes + " H2 Fuel Cell",
        bus0=spatial.nodes + " H2",
        bus1=spatial.nodes,
        p_nom_extendable=True,
        carrier="H2 Fuel Cell",
        efficiency=costs.at["fuel cell", "efficiency"],
        # NB: fixed cost is per MWel
        capital_cost=costs.at["fuel cell", "fixed"]
        * costs.at["fuel cell", "efficiency"],
        lifetime=costs.at["fuel cell", "lifetime"],
    )

    cavern_nodes = pd.DataFrame()

    if snakemake.config["sector"]["hydrogen"]["underground_storage"]:
        if snakemake.config["custom_data"]["h2_underground"]:
            custom_cavern = pd.read_csv(
                os.path.join(
                    BASE_DIR,
                    "data/custom/h2_underground_{0}_{1}.csv".format(
                        demand_sc, investment_year
                    ),
                )
            )
            # countries = n.buses.country.unique().to_list()
            countries = snakemake.config["countries"]
            custom_cavern = custom_cavern[custom_cavern.country.isin(countries)]

            cavern_nodes = n.buses[n.buses.country.isin(custom_cavern.country)]

            h2_pot = custom_cavern.set_index("id_region")["storage_cap_MWh"]

            h2_capital_cost = costs.at["hydrogen storage underground", "fixed"]

            # h2_pot.index = cavern_nodes.index

            # n.add("Carrier", "H2 UHS")

            n.madd(
                "Bus",
                nodes + " H2 UHS",
                location=nodes,
                carrier="H2 UHS",
                x=n.buses.loc[list(nodes)].x.values,
                y=n.buses.loc[list(nodes)].y.values,
            )

            n.madd(
                "Store",
                cavern_nodes.index + " H2 UHS",
                bus=cavern_nodes.index + " H2 UHS",
                e_nom_extendable=True,
                e_nom_max=h2_pot.values,
                e_cyclic=True,
                carrier="H2 UHS",
                capital_cost=h2_capital_cost,
            )

            n.madd(
                "Link",
                nodes + " H2 UHS charger",
                bus0=nodes + " H2",
                bus1=nodes + " H2 UHS",
                carrier="H2 UHS charger",
                # efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
                # capital_cost=costs.at["battery inverter", "fixed"],
                p_nom_extendable=True,
                # lifetime=costs.at["battery inverter", "lifetime"],
            )

            n.madd(
                "Link",
                nodes + " H2 UHS discharger",
                bus0=nodes + " H2 UHS",
                bus1=nodes + " H2",
                carrier="H2 UHS discharger",
                efficiency=1,
                # capital_cost=costs.at["battery inverter", "fixed"],
                p_nom_extendable=True,
                # lifetime=costs.at["battery inverter", "lifetime"],
            )

        else:
            h2_salt_cavern_potential = pd.read_csv(
                snakemake.input.h2_cavern, index_col=0
            ).squeeze()
            h2_cavern_ct = h2_salt_cavern_potential[~h2_salt_cavern_potential.isna()]
            cavern_nodes = n.buses[n.buses.country.isin(h2_cavern_ct.index)]

            h2_capital_cost = costs.at["hydrogen storage underground", "fixed"]

            # assumptions: weight storage potential in a country by population
            # TODO: fix with real geographic potentials
            # convert TWh to MWh with 1e6
            h2_pot = h2_cavern_ct.loc[cavern_nodes.country]
            h2_pot.index = cavern_nodes.index

            # distribute underground potential equally over all nodes #TODO change with real data
            s = pd.Series(h2_pot.index, index=h2_pot.index)
            country_codes = s.str[:2]
            code_counts = country_codes.value_counts()
            fractions = country_codes.map(code_counts).rdiv(1)
            h2_pot = h2_pot * fractions * 1e6

            # n.add("Carrier", "H2 UHS")

            n.madd(
                "Bus",
                nodes + " H2 UHS",
                location=nodes,
                carrier="H2 UHS",
                x=n.buses.loc[list(nodes)].x.values,
                y=n.buses.loc[list(nodes)].y.values,
            )

            n.madd(
                "Store",
                cavern_nodes.index + " H2 UHS",
                bus=cavern_nodes.index + " H2 UHS",
                e_nom_extendable=True,
                e_nom_max=h2_pot.values,
                e_cyclic=True,
                carrier="H2 UHS",
                capital_cost=h2_capital_cost,
            )

            n.madd(
                "Link",
                nodes + " H2 UHS charger",
                bus0=nodes,
                bus1=nodes + " H2 UHS",
                carrier="H2 UHS charger",
                # efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
                capital_cost=0,
                p_nom_extendable=True,
                # lifetime=costs.at["battery inverter", "lifetime"],
            )

            n.madd(
                "Link",
                nodes + " H2 UHS discharger",
                bus0=nodes,
                bus1=nodes + " H2 UHS",
                carrier="H2 UHS discharger",
                efficiency=1,
                capital_cost=0,
                p_nom_extendable=True,
                # lifetime=costs.at["battery inverter", "lifetime"],
            )

    # hydrogen stored overground (where not already underground)
    h2_capital_cost = costs.at[
        "hydrogen storage tank type 1 including compressor", "fixed"
    ]
    nodes_overground = nodes
    n.madd(
        "Store",
        nodes_overground + " H2 Store Tank",
        bus=nodes_overground + " H2",
        e_nom_extendable=True,
        p_nom_extendable=True,
        p_nom=1.0,              # MW baseline (non-binding, but not zero/NaN)
        e_cyclic=True,
        carrier="H2 Store Tank",
        capital_cost=h2_capital_cost,
    )

    # Hydrogen network:
    # -----------------
    def add_links_repurposed_H2_pipelines():
        n.madd(
            "Link",
            h2_links.index + " repurposed",
            bus0=h2_links.bus0.values + " H2",
            bus1=h2_links.bus1.values + " H2",
            p_min_pu=-1,
            p_nom_extendable=True,
            p_nom_max=h2_links.capacity.values
            * 0.8,  # https://gasforclimate2050.eu/wp-content/uploads/2020/07/2020_European-Hydrogen-Backbone_Report.pdf
            length=h2_links.length.values,
            capital_cost=costs.at["H2 (g) pipeline repurposed", "fixed"]
            * h2_links.length.values,
            carrier="H2 pipeline repurposed",
            lifetime=costs.at["H2 (g) pipeline repurposed", "lifetime"],
        )

    def add_links_new_H2_pipelines():
        n.madd(
            "Link",
            h2_links.index,
            bus0=h2_links.bus0.values + " H2",
            bus1=h2_links.bus1.values + " H2",
            p_min_pu=-1,
            p_nom_extendable=True,
            length=h2_links.length.values,
            capital_cost=costs.at["H2 (g) pipeline", "fixed"] * h2_links.length.values,
            carrier="H2 pipeline",
            lifetime=costs.at["H2 (g) pipeline", "lifetime"],
        )

    def add_links_elec_routing_new_H2_pipelines():
        attrs = ["bus0", "bus1", "length"]
        h2_links = pd.DataFrame(columns=attrs)

        candidates = pd.concat(
            {
                "lines": n.lines[attrs],
                "links": n.links.loc[n.links.carrier == "DC", attrs],
            }
        )

        for candidate in candidates.index:
            buses = [
                candidates.at[candidate, "bus0"],
                candidates.at[candidate, "bus1"],
            ]
            buses.sort()
            name = f"H2 pipeline {buses[0]} -> {buses[1]}"
            if name not in h2_links.index:
                h2_links.at[name, "bus0"] = buses[0]
                h2_links.at[name, "bus1"] = buses[1]
                h2_links.at[name, "length"] = candidates.at[candidate, "length"]

        n.madd(
            "Link",
            h2_links.index,
            bus0=h2_links.bus0.values + " H2",
            bus1=h2_links.bus1.values + " H2",
            p_min_pu=-1,
            p_nom_extendable=True,
            length=h2_links.length.values,
            capital_cost=costs.at["H2 (g) pipeline", "fixed"] * h2_links.length.values,
            carrier="H2 pipeline",
            lifetime=costs.at["H2 (g) pipeline", "lifetime"],
        )

    # Add H2 Links:
    if snakemake.config["sector"]["hydrogen"]["network"]:
        h2_links = pd.read_csv(snakemake.input.pipelines)

        # Order buses to detect equal pairs for bidirectional pipelines
        # buses_ordered = h2_links.apply(lambda p: sorted([p.bus0, p.bus1]), axis=1)

        # Appending string for carrier specification '_AC'
        # h2_links["bus0"] = buses_ordered.str[0] + "_AC"
        # h2_links["bus1"] = buses_ordered.str[1] + "_AC"

        # Create index column
        h2_links["buses_idx"] = (
            "H2 pipeline " + h2_links["bus0"] + " -> " + h2_links["bus1"]
        )

        # Aggregate pipelines applying mean on length and sum on capacities
        h2_links = h2_links.groupby("buses_idx").agg(
            {"bus0": "first", "bus1": "first", "length": "mean", "capacity": "sum"}
        )

        if len(h2_links) > 0:
            if snakemake.config["sector"]["hydrogen"]["gas_network_repurposing"]:
                add_links_repurposed_H2_pipelines()
            if snakemake.config["sector"]["hydrogen"]["network_routes"] == "greenfield":
                add_links_elec_routing_new_H2_pipelines()
            else:
                add_links_new_H2_pipelines()
        else:
            print(
                "No existing gas network; applying greenfield for H2 network"
            )  # TODO change to logger.info
            add_links_elec_routing_new_H2_pipelines()

        if snakemake.config["sector"]["hydrogen"]["hydrogen_colors"]:
            nuclear_gens_bus = n.generators[
                n.generators.carrier == "nuclear"
            ].bus.values
            buses_with_nuclear = n.buses.loc[nuclear_gens_bus]
            buses_with_nuclear_ind = n.buses.loc[nuclear_gens_bus].index

            # nn.add("Carrier", "nuclear electricity")
            # nn.add("Carrier", "pink H2")

            n.madd(
                "Bus",
                nuclear_gens_bus + " nuclear electricity",
                location=buses_with_nuclear_ind,
                carrier="nuclear electricity",
                x=buses_with_nuclear.x.values,
                y=buses_with_nuclear.y.values,
            )

            n.madd(
                "Bus",
                nuclear_gens_bus + " pink H2",
                location=buses_with_nuclear_ind,
                carrier="pink H2",
                x=buses_with_nuclear.x.values,
                y=buses_with_nuclear.y.values,
            )

            n.generators.loc[n.generators.carrier == "nuclear", "bus"] = (
                n.generators.loc[n.generators.carrier == "nuclear", "bus"]
                + " nuclear electricity"
            )

            n.madd(
                "Link",
                buses_with_nuclear_ind + " nuclear-to-grid",
                bus0=buses_with_nuclear_ind + " nuclear electricity",
                bus1=buses_with_nuclear_ind,
                carrier="nuclear-to-grid",
                capital_cost=0,
                p_nom_extendable=True,
                # lifetime=costs.at["battery inverter", "lifetime"],
            )

            n.madd(
                "Link",
                buses_with_nuclear_ind + " high-temp electrolysis",
                bus0=buses_with_nuclear_ind + " nuclear electricity",
                bus1=buses_with_nuclear_ind + " pink H2",
                carrier="high-temp electrolysis",
                # capital_cost=0,
                p_nom_extendable=True,
                efficiency=costs.at["electrolysis", "efficiency"] + 0.1,
                capital_cost=costs.at["electrolysis", "fixed"]
                + costs.at["electrolysis", "fixed"] * 0.1,
                lifetime=costs.at["electrolysis", "lifetime"],
            )

            n.madd(
                "Link",
                buses_with_nuclear_ind + " pink H2",
                bus0=buses_with_nuclear_ind + " pink H2",
                bus1=buses_with_nuclear_ind + " H2",
                carrier="pink H2",
                # efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
                capital_cost=0,
                p_nom_extendable=True,
                # lifetime=costs.at["battery inverter", "lifetime"],
            )


def define_spatial(nodes, options):
    """
    Namespace for spatial.

    Parameters
    ----------
    nodes : list-like
    """

    global spatial

    spatial.nodes = nodes

    print("\n===== SPATIAL DEBUG START =====")
    print("Total base nodes:", len(nodes))
    print("Sample nodes:", list(nodes)[:5])

    # biomass
    spatial.biomass = SimpleNamespace()

    if options["biomass_transport"]:
        spatial.biomass.nodes = nodes + " solid biomass"
        spatial.biomass.locations = nodes
        spatial.biomass.industry = nodes + " solid biomass for industry"
        spatial.biomass.industry_cc = nodes + " solid biomass for industry CC"
    else:
        spatial.biomass.nodes = ["Earth solid biomass"]
        spatial.biomass.locations = ["Earth"]
        spatial.biomass.industry = ["solid biomass for industry"]
        spatial.biomass.industry_cc = ["solid biomass for industry CC"]

    spatial.biomass.df = pd.DataFrame(vars(spatial.biomass), index=nodes)

    print("\n[BIOMASS]")
    print("nodes:", len(spatial.biomass.nodes))
    print("locations:", len(spatial.biomass.locations))
    print("industry:", len(spatial.biomass.industry))
    print("industry_cc:", len(spatial.biomass.industry_cc))
    print("sample nodes:", spatial.biomass.nodes[:3])
    print("df shape:", spatial.biomass.df.shape)

    # co2
    spatial.co2 = SimpleNamespace()

    if options["co2_network"]:
        spatial.co2.nodes = nodes + " co2 stored"
        spatial.co2.locations = nodes
        spatial.co2.vents = nodes + " co2 vent"
    else:
        spatial.co2.nodes = ["co2 stored"]
        spatial.co2.locations = ["Earth"]
        spatial.co2.vents = ["co2 vent"]

    spatial.co2.df = pd.DataFrame(vars(spatial.co2), index=nodes)

    print("\n[CO2]")
    print("nodes:", len(spatial.co2.nodes))
    print("locations:", len(spatial.co2.locations))
    print("vents:", len(spatial.co2.vents))
    print("sample nodes:", spatial.co2.nodes[:3])
    print("df shape:", spatial.co2.df.shape)

    # oil
    spatial.oil = SimpleNamespace()

    if options["oil"]["spatial_oil"]:
        spatial.oil.nodes = nodes + " oil"
        spatial.oil.locations = nodes
    else:
        spatial.oil.nodes = ["Earth oil"]
        spatial.oil.locations = ["Earth"]

    print("\n[OIL]")
    print("nodes:", len(spatial.oil.nodes))
    print("locations:", len(spatial.oil.locations))
    print("sample nodes:", spatial.oil.nodes[:3])

    # gas
    spatial.gas = SimpleNamespace()

    if options["gas"]["spatial_gas"]:
        spatial.gas.nodes = nodes + " gas"
        spatial.gas.locations = nodes
        spatial.gas.biogas = nodes + " biogas"
        spatial.gas.industry = nodes + " gas for industry"
        if options["cc"]:
            spatial.gas.industry_cc = nodes + " gas for industry CC"
        spatial.gas.biogas_to_gas = nodes + " biogas to gas"
    else:
        spatial.gas.nodes = ["Earth gas"]
        spatial.gas.locations = ["Earth"]
        spatial.gas.biogas = ["Earth biogas"]
        spatial.gas.industry = ["gas for industry"]
        if options["cc"]:
            spatial.gas.industry_cc = ["gas for industry CC"]
        spatial.gas.biogas_to_gas = ["Earth biogas to gas"]

    spatial.gas.df = pd.DataFrame(vars(spatial.gas), index=spatial.nodes)

    print("\n[GAS]")
    print("nodes:", len(spatial.gas.nodes))
    print("locations:", len(spatial.gas.locations))
    print("biogas:", len(spatial.gas.biogas))
    print("industry:", len(spatial.gas.industry))
    print("sample nodes:", spatial.gas.nodes[:3])
    print("df shape:", spatial.gas.df.shape)

    # coal
    spatial.coal = SimpleNamespace()

    if options["coal"]["spatial_coal"]:
        spatial.coal.nodes = nodes + " coal"
        spatial.coal.locations = nodes
        spatial.coal.industry = nodes + " coal for industry"
    else:
        spatial.coal.nodes = ["Earth coal"]
        spatial.coal.locations = ["Earth"]
        spatial.coal.industry = ["Earth coal for industry"]

    spatial.coal.df = pd.DataFrame(vars(spatial.coal), index=spatial.nodes)

    print("\n[COAL]")
    print("nodes:", len(spatial.coal.nodes))
    print("locations:", len(spatial.coal.locations))
    print("industry:", len(spatial.coal.industry))
    print("sample nodes:", spatial.coal.nodes[:3])
    print("df shape:", spatial.coal.df.shape)

    # lignite
    spatial.lignite = SimpleNamespace()

    if options["lignite"]["spatial_lignite"]:
        spatial.lignite.nodes = nodes + " lignite"
        spatial.lignite.locations = nodes
    else:
        spatial.lignite.nodes = ["Earth lignite"]
        spatial.lignite.locations = ["Earth"]

    spatial.lignite.df = pd.DataFrame(vars(spatial.lignite), index=spatial.nodes)

    print("\n[LIGNITE]")
    print("nodes:", len(spatial.lignite.nodes))
    print("locations:", len(spatial.lignite.locations))
    print("sample nodes:", spatial.lignite.nodes[:3])
    print("df shape:", spatial.lignite.df.shape)

    print("\n===== SPATIAL DEBUG END =====\n")

    return spatial

def add_biomass(n, costs):
    """
    Biomass + biogas potentials as annual fuel Stores (MWh/a) distributed nodally,
    plus:
      - solid biomass -> electricity conversion ("biomass EOP") on electricity buses
      - optional biogas upgrading (biogas -> gas + CO2)
      - optional biomass transport
      - optional CHP (+ optional CHP CC)

    Includes debug prints to verify:
      - potentials, pathway fraction
      - created buses/stores/links counts
      - total e_nom checks
      - EOP link caps sanity
    """
    logger.info("adding biomass")

    # -----------------------------
    # 0) read config + apply pathway
    # -----------------------------
    biomass_pot = float(snakemake.config["sector"]["solid_biomass_potential"]) * 1e6  # TWh -> MWh
    biogas_pot  = float(snakemake.config["sector"]["biogas_potential"]) * 1e6        # TWh -> MWh

    sched = snakemake.config["sector"].get("bio_pathway", {})  # {year: frac}
    frac = sched.get(investment_year, sched.get(str(investment_year), 1.0))
    frac = max(0.0, min(1.0, float(frac)))

    biomass_pot *= frac
    biogas_pot  *= frac

    print(f"[biomass] bio_pathway frac @ {investment_year} = {frac}")
    print(f"[biomass] total potentials [MWh/a]: solid={biomass_pot:.3f}, biogas={biogas_pot:.3f}")

    # -----------------------------
    # 1) identify electricity buses
    # -----------------------------
    bus_car = n.buses["carrier"].astype("string").fillna("")
    elec_buses = n.buses.index[bus_car.isin(["AC", "DC"])].astype(str)

    print(f"[biomass] elec buses (AC/DC) count: {len(elec_buses)}")
    if len(elec_buses) == 0:
        print("[biomass][WARN] no AC/DC buses found -> biomass buses will be empty unless referenced by links")

    # build nodal biomass buses on electricity buses
    biomass_nodes = pd.Index(elec_buses) + " solid biomass"

    # defensive: include any " solid biomass" buses referenced in existing links
    link_bus_cols = [c for c in ["bus0", "bus1", "bus2", "bus3", "bus4"] if c in n.links.columns]
    if link_bus_cols and not n.links.empty:
        referenced = pd.Index(
            pd.unique(pd.concat([n.links[c].astype("string") for c in link_bus_cols], axis=0))
        ).dropna()
        referenced = referenced[referenced.astype(str).str.endswith(" solid biomass")]
        if len(referenced):
            print(f"[biomass] found referenced solid biomass buses in links: {len(referenced)}")
        biomass_nodes = biomass_nodes.union(referenced.astype(str))

    # biogas buses (keep your existing spatial logic)
    biogas_nodes = pd.Index(spatial.gas.biogas).astype(str)
    print(f"[biomass] biogas buses count (spatial.gas.biogas): {len(biogas_nodes)}")

    # -----------------------------
    # 2) carriers (idempotent)
    # -----------------------------
    if "biogas" not in n.carriers.index:
        n.add("Carrier", "biogas")
    if "solid biomass" not in n.carriers.index:
        n.add("Carrier", "solid biomass")
    if "biomass EOP" not in n.carriers.index:
        n.add("Carrier", "biomass EOP")

    # -----------------------------
    # 3) add missing buses
    # -----------------------------
    # solid biomass buses
    missing_biomass_buses = biomass_nodes.difference(n.buses.index.astype(str))
    print(f"[biomass] missing solid biomass buses to add: {len(missing_biomass_buses)}")
    if len(missing_biomass_buses):
        base = missing_biomass_buses.str.replace(r" solid biomass$", "", regex=True).astype(str)
        base_buses = n.buses.reindex(base)

        n.madd(
            "Bus",
            missing_biomass_buses,
            location=base,
            carrier="solid biomass",
            x=base_buses.x.fillna(0.0).values,
            y=base_buses.y.fillna(0.0).values,
        )

    # biogas buses
    missing_biogas_buses = biogas_nodes.difference(n.buses.index.astype(str))
    print(f"[biomass] missing biogas buses to add: {len(missing_biogas_buses)}")
    if len(missing_biogas_buses):
        n.madd(
            "Bus",
            missing_biogas_buses,
            location=getattr(spatial.biomass, "locations", None),
            carrier="biogas",
        )

    # -----------------------------
    # 4) distribute annual potentials across nodes
    # -----------------------------
    def as_nodal_energy(total_mwh_per_year, nodes):
        nodes = pd.Index(nodes).astype(str)
        if len(nodes) == 0:
            return pd.Series(dtype=float, index=nodes)
        return pd.Series(float(total_mwh_per_year) / len(nodes), index=nodes)

    biomass_e_nom = as_nodal_energy(biomass_pot, biomass_nodes)  # MWh/a per biomass bus
    biogas_e_nom  = as_nodal_energy(biogas_pot, biogas_nodes)    # MWh/a per biogas bus

    print(f"[biomass] biomass_nodes count used for distribution: {len(biomass_nodes)}")
    print(f"[biomass] biogas_nodes  count used for distribution: {len(biogas_nodes)}")

    # -----------------------------
    # 5) add Stores (fuel availability)
    # -----------------------------
    # solid biomass stores
    missing_biomass_stores = biomass_nodes.difference(n.stores.index.astype(str))
    print(f"[biomass] missing solid biomass stores to add: {len(missing_biomass_stores)}")
    if len(missing_biomass_stores):
        s = biomass_e_nom.reindex(missing_biomass_stores).fillna(0.0)
        if "solid biomass" not in costs.index:
            print("[biomass][WARN] 'solid biomass' not in costs.index; setting marginal_cost=0.0")
            mc = 0.0
        else:
            mc = float(costs.at["solid biomass", "fuel"])

        n.madd(
            "Store",
            missing_biomass_stores,
            bus=missing_biomass_stores,
            carrier="solid biomass",
            e_nom=s.values,
            e_initial=s.values,
            e_cyclic=False,
            e_initial_per_period=True,
            marginal_cost=mc,
        )
    n.stores.loc[n.stores.carrier == "solid biomass", "e_min_pu"] = 0.0

    # biogas stores
    missing_biogas_stores = biogas_nodes.difference(n.stores.index.astype(str))
    print(f"[biomass] missing biogas stores to add: {len(missing_biogas_stores)}")
    if len(missing_biogas_stores):
        s = biogas_e_nom.reindex(missing_biogas_stores).fillna(0.0)
        if "biogas" not in costs.index:
            print("[biomass][WARN] 'biogas' not in costs.index; setting marginal_cost=0.0")
            mc = 0.0
        else:
            mc = float(costs.at["biogas", "fuel"])

        n.madd(
            "Store",
            missing_biogas_stores,
            bus=missing_biogas_stores,
            carrier="biogas",
            e_nom=s.values,
            e_initial=s.values,
            e_cyclic=False,
            e_initial_per_period=True,
            marginal_cost=mc,
        )
    n.stores.loc[n.stores.carrier == "biogas", "e_min_pu"] = 0.0

    print(
        "[biomass] CHECK total solid biomass e_nom [MWh/a]:",
        float(n.stores.loc[n.stores.carrier == "solid biomass", "e_nom"].sum()),
    )
    print(
        "[biomass] CHECK total biogas e_nom [MWh/a]:",
        float(n.stores.loc[n.stores.carrier == "biogas", "e_nom"].sum()),
    )

    # -----------------------------
    # 6) biomass EOP (solid biomass -> electricity)  [FIXED]
    # -----------------------------
    H_yr = float(n.snapshot_weightings.generators.sum())
    print(f"[biomass] H_yr (snapshot_weightings.generators.sum): {H_yr}")

    if H_yr <= 0:
        print("[biomass][WARN] H_yr <= 0 -> setting EOP caps to 0.0")
        pmax_by_biomass_bus = biomass_e_nom.copy() * 0.0
    else:
        pmax_by_biomass_bus = (biomass_e_nom / H_yr).fillna(0.0)  # MW_fuel cap per biomass bus

   # only build EOP for biomass buses whose base bus exists as an electricity bus
    elec_from_biomass = biomass_nodes.str.replace(r" solid biomass$", "", regex=True).astype(str)

    valid_mask = elec_from_biomass.isin(n.buses.index.astype(str))
    if (~valid_mask).any():
        bad = elec_from_biomass[~valid_mask]
        print(f"[biomass][WARN] skipping EOP for non-existent base buses: {len(bad)}")
        print("[biomass][WARN] sample:", bad[:5].tolist())

    biomass_nodes_eop = biomass_nodes[valid_mask]
    elec_from_biomass_eop = elec_from_biomass[valid_mask]
    eop_link_names = elec_from_biomass_eop + " biomass EOP"

    p_nom_max_eop = pd.Series(
        pmax_by_biomass_bus.reindex(biomass_nodes_eop).fillna(0.0).values,
        index=eop_link_names,
    )

    missing_eop_links = eop_link_names.difference(n.links.index.astype(str))
    print(f"[biomass] biomass EOP links to add: {len(missing_eop_links)} (total desired: {len(eop_link_names)})")

    if len(missing_eop_links):
        ac = missing_eop_links.str.replace(r" biomass EOP$", "", regex=True).astype(str)

        # basic existence checks (debug)
        b0 = ac + " solid biomass"
        b1 = ac
        b0_missing = b0.difference(n.buses.index.astype(str))
        b1_missing = b1.difference(n.buses.index.astype(str))
        if len(b0_missing) or len(b1_missing):
            print(f"[biomass][WARN] EOP bus existence issue: missing bus0={len(b0_missing)}, missing bus1={len(b1_missing)}")
            if len(b0_missing):
                print("[biomass][WARN] sample missing bus0:", list(b0_missing[:5]))
            if len(b1_missing):
                print("[biomass][WARN] sample missing bus1:", list(b1_missing[:5]))

        if "biomass EOP" not in costs.index:
            print("[biomass][WARN] 'biomass EOP' not in costs.index; using zero costs + eff=1.0")
            eff = 1.0
            capex = 0.0
            vom = 0.0
            life = 25.0
        else:
            eff  = float(costs.at["biomass EOP", "efficiency"])
            capex = float(costs.at["biomass EOP", "fixed"])
            vom  = float(costs.at["biomass EOP", "VOM"])
            life = float(costs.at["biomass EOP", "lifetime"])

        n.madd(
            "Link",
            missing_eop_links,
            bus0=b0,
            bus1=b1,
            carrier="biomass EOP",
            p_nom_extendable=True,
            p_nom_max=p_nom_max_eop.reindex(missing_eop_links).fillna(0.0).values,
            efficiency=eff,
            capital_cost=capex,
            marginal_cost=vom,
            lifetime=life,
        )
    else:
        # ensure caps are set even if links already exist
        n.links.loc[eop_link_names, "p_nom_max"] = p_nom_max_eop.reindex(eop_link_names).values

    # debug: verify EOP presence + caps
    n_eop = int((n.links.carrier == "biomass EOP").sum()) if not n.links.empty else 0
    print(f"[biomass] biomass EOP links in network now: {n_eop}")
    if n_eop:
        caps = pd.to_numeric(n.links.loc[n.links.carrier == "biomass EOP", "p_nom_max"], errors="coerce").fillna(0.0)
        print(f"[biomass] biomass EOP p_nom_max: min={caps.min():.6f}, mean={caps.mean():.6f}, max={caps.max():.6f}")

    # -----------------------------
    # 7) cap legacy "<bus> biomass" links (optional)
    # -----------------------------
    legacy = n.links.index[n.links.index.astype(str).str.contains(r" biomass$", regex=True)]
    if len(legacy):
        legacy = pd.Index(legacy).astype(str)
        legacy_bus = legacy.str.replace(r" biomass$", "", regex=True).astype(str)
        legacy_biomass_bus = legacy_bus + " solid biomass"
        pmax_legacy = (biomass_e_nom.reindex(legacy_biomass_bus).fillna(0.0) / max(H_yr, 1.0)).values
        n.links.loc[legacy, "p_nom_max"] = pmax_legacy
        print(f"[biomass] capped legacy '* biomass' links: {len(legacy)}")

    # -----------------------------
    # 8) biogas upgrading (biogas -> gas + CO2)
    # -----------------------------
    if "biogas upgrading" in costs.index or "biogas to gas" in costs.index:
        key = "biogas upgrading" if "biogas upgrading" in costs.index else "biogas to gas"
        print(f"[biomass] adding biogas upgrading links using costs row: '{key}'")

        n.madd(
            "Link",
            spatial.gas.biogas_to_gas,
            bus0=spatial.gas.biogas,
            bus1=spatial.gas.nodes,
            bus2="co2 atmosphere",
            carrier="biogas to gas",
            capital_cost=float(costs.loc[key, "fixed"]),
            marginal_cost=float(costs.loc[key, "VOM"]),
            efficiency2=-float(costs.at["gas", "CO2 intensity"]),
            p_nom_extendable=True,
        )
    else:
        print("[biomass] biogas upgrading not added (no matching costs row)")

    # -----------------------------
    # 9) biomass transport
    # -----------------------------
    if options.get("biomass_transport", False):
        print("[biomass] biomass transport enabled")

        transport_costs = pd.read_csv(
            snakemake.input.biomass_transport_costs,
            index_col=0,
            keep_default_na=False,
        ).squeeze()

        biomass_transport = create_network_topology(n, "biomass transport ", bidirectional=False)

        bus0_costs = biomass_transport.bus0.apply(
            lambda x: transport_costs.get(
                str(x)[:2], snakemake.config["sector"]["biomass_transport_default_cost"]
            )
        )
        bus1_costs = biomass_transport.bus1.apply(
            lambda x: transport_costs.get(
                str(x)[:2], snakemake.config["sector"]["biomass_transport_default_cost"]
            )
        )
        biomass_transport["costs"] = pd.concat([bus0_costs, bus1_costs], axis=1).mean(axis=1)

        n.madd(
            "Link",
            biomass_transport.index,
            bus0=biomass_transport.bus0.astype(str) + " solid biomass",
            bus1=biomass_transport.bus1.astype(str) + " solid biomass",
            p_nom_extendable=True,
            length=biomass_transport.length.values,
            marginal_cost=biomass_transport.costs * biomass_transport.length.values,
            capital_cost=1,
            carrier="solid biomass transport",
        )

        print(f"[biomass] biomass transport links added: {len(biomass_transport.index)}")
    else:
        print("[biomass] biomass transport disabled")

    # -----------------------------
    # 10) CHP (optional)
    # -----------------------------
    urban_central = n.buses.index[n.buses.carrier == "urban central heat"]
    if not urban_central.empty and options.get("chp", False):
        print(f"[biomass] CHP enabled; urban central heat buses: {len(urban_central)}")

        urban_central = urban_central.str[: -len(" urban central heat")]
        key = "central solid biomass CHP"

        n.madd(
            "Link",
            urban_central + " urban central solid biomass CHP",
            bus0=spatial.biomass.df.loc[urban_central, "nodes"].values,
            bus1=urban_central,
            bus2=urban_central + " urban central heat",
            carrier="urban central solid biomass CHP",
            p_nom_extendable=True,
            capital_cost=float(costs.at[key, "fixed"]) * float(costs.at[key, "efficiency"]),
            marginal_cost=float(costs.at[key, "VOM"]),
            efficiency=float(costs.at[key, "efficiency"]),
            efficiency2=float(costs.at[key, "efficiency-heat"]),
            lifetime=float(costs.at[key, "lifetime"]),
        )

        eff_chp = float(costs.at[key, "efficiency"])
        scalar_cap = (biomass_pot / max(1, len(biomass_nodes))) / (max(H_yr, 1.0) * eff_chp) if H_yr > 0 else 0.0

        chp_links = n.links.index[n.links.carrier == "urban central solid biomass CHP"]
        if len(chp_links):
            n.links.loc[chp_links, "p_nom_max"] = scalar_cap
            print(f"[biomass] CHP links capped (p_nom_max={scalar_cap:.6f}) on {len(chp_links)} links")

        if snakemake.config["sector"].get("cc", False):
            print("[biomass] CHP CC enabled")

            n.madd(
                "Link",
                urban_central + " urban central solid biomass CHP CC",
                bus0=spatial.biomass.df.loc[urban_central, "nodes"] + " solid biomass",
                bus1=urban_central,
                bus2=urban_central + " urban central heat",
                bus3="co2 atmosphere",
                bus4=spatial.co2.df.loc[urban_central, "nodes"].values,
                carrier="urban central solid biomass CHP CC",
                p_nom_extendable=True,
                capital_cost=float(costs.at[key, "fixed"]) * float(costs.at[key, "efficiency"])
                + float(costs.at["biomass CHP capture", "fixed"]) * float(costs.at["solid biomass", "CO2 intensity"]),
                marginal_cost=float(costs.at[key, "VOM"]),
                efficiency=float(costs.at[key, "efficiency"])
                - float(costs.at["solid biomass", "CO2 intensity"])
                * (
                    float(costs.at["biomass CHP capture", "electricity-input"])
                    + float(costs.at["biomass CHP capture", "compression-electricity-input"])
                ),
                efficiency2=float(costs.at[key, "efficiency-heat"])
                + float(costs.at["solid biomass", "CO2 intensity"])
                * (
                    float(costs.at["biomass CHP capture", "heat-output"])
                    + float(costs.at["biomass CHP capture", "compression-heat-output"])
                    - float(costs.at["biomass CHP capture", "heat-input"])
                ),
                efficiency3=-float(costs.at["solid biomass", "CO2 intensity"])
                * float(costs.at["biomass CHP capture", "capture_rate"]),
                efficiency4=float(costs.at["solid biomass", "CO2 intensity"])
                * float(costs.at["biomass CHP capture", "capture_rate"]),
                lifetime=float(costs.at[key, "lifetime"]),
            )

            chpcc_links = n.links.index[n.links.carrier == "urban central solid biomass CHP CC"]
            if len(chpcc_links):
                n.links.loc[chpcc_links, "p_nom_max"] = scalar_cap
                print(f"[biomass] CHP CC links capped (p_nom_max={scalar_cap:.6f}) on {len(chpcc_links)} links")
    else:
        if options.get("chp", False):
            print("[biomass][WARN] CHP enabled in options but no 'urban central heat' buses found")
        else:
            print("[biomass] CHP disabled")

    # final summary debug
    print("[biomass] SUMMARY carriers present:", [c for c in ["solid biomass", "biogas", "biomass EOP"] if c in n.carriers.index])
    print("[biomass] SUMMARY counts:",
          "buses(solid biomass)=", int((n.buses.carrier == "solid biomass").sum()) if not n.buses.empty else 0,
          "buses(biogas)=", int((n.buses.carrier == "biogas").sum()) if not n.buses.empty else 0,
          "stores(solid biomass)=", int((n.stores.carrier == "solid biomass").sum()) if not n.stores.empty else 0,
          "stores(biogas)=", int((n.stores.carrier == "biogas").sum()) if not n.stores.empty else 0,
          "links(biomass EOP)=", int((n.links.carrier == "biomass EOP").sum()) if not n.links.empty else 0)
    
def co2_cap_from_config(cfg, year):
    el = cfg["electricity"]
    base = float(el.get("co2limit", 0.0))  # cast even if "1e+9" was a string

    use_rel = el.get("use_relative_targets", False)
    if isinstance(use_rel, str):
        use_rel = use_rel.strip().lower() in ("1", "true", "yes", "y")
    if not use_rel:
        return base

    # keys/values may be strings in YAML → cast both
    rel = {int(k): float(v) for k, v in el.get("co2_relative_targets", {}).items()}
    if not rel:
        return base

    y = int(year)
    years = sorted(rel)
    # step behavior: use the last target <= year (fallback to smallest key)
    k = max([yy for yy in years if yy <= y], default=years[0])
    frac = rel[k]
    print(f"CO2 cap for {y}: {base * frac}")
    return base * frac

def _debug_co2_budget(n, cap, tag=""):
    """
    prints how much *fixed* CO2 is being injected into co2 atmosphere via Loads.
    if this exceeds the capped atmosphere store, you're infeasible.
    """
    import numpy as np

    if "co2 atmosphere" not in n.buses.index:
        print("[co2 debug] no co2 atmosphere bus.")
        return

    co2_loads = n.loads[n.loads.bus == "co2 atmosphere"] if not n.loads.empty else n.loads.iloc[0:0]
    if co2_loads.empty:
        print("[co2 debug] no loads on co2 atmosphere bus.")
        return

    # snapshot-weighted energy injected (MWh equivalent) over the modeled horizon
    w = n.snapshot_weightings.generators.reindex(n.snapshots).fillna(1.0)
    inj = 0.0
    for name in co2_loads.index:
        # p_set is time series (MW). Energy = sum(p_set * weight_hours)
        if name in n.loads_t.p_set.columns:
            s = n.loads_t.p_set[name].reindex(n.snapshots).fillna(0.0)
            inj += float((s * w).sum())

    # sign convention: your CO2 loads are often negative for "injection".
    # we care about net increase of CO2 in atmosphere store -> take negative part magnitude.
    injected_positive = max(0.0, -inj)

    print(f"\n[co2 debug]{' '+tag if tag else ''}")
    print(f"  cap (store e_nom): {float(cap):.4g}")
    print(f"  fixed injected into atmosphere (weighted): {injected_positive:.4g}")
    if injected_positive > float(cap) + 1e-6:
        print("  >>> INFEASIBLE LIKELY: fixed CO2 injection exceeds atmosphere cap")

def add_co2(n, costs,Nyears):
    "add carbon carrier, it's networks and storage units"

    # minus sign because opposite to how fossil fuels used:
    # CH4 burning puts CH4 down, atmosphere up
    n.add("Carrier", "co2", co2_emissions=-1.0)

    # this tracks CO2 in the atmosphere
    n.add(
        "Bus",
        "co2 atmosphere",
        location="Earth",  # TODO Ignoed by pypsa check
        carrier="co2",
    )

    # can also be negative
    n.add(
        "Store",
        "co2 atmosphere",
        e_nom_extendable=True,
        e_min_pu=-1,
        carrier="co2",
        bus="co2 atmosphere",
    )

    # this tracks CO2 stored, e.g. underground
    n.madd(
        "Bus",
        spatial.co2.nodes,
        location=spatial.co2.locations,
        carrier="co2 stored",
        # x=spatial.co2.x[0],
        # y=spatial.co2.y[0],
    )
    """
    co2_stored_x = n.buses.filter(like="co2 stored", axis=0).loc[:, "x"]
    co2_stored_y = n.buses.loc[n.buses[n.buses.carrier == "co2
    stored"].location].y.

    n.buses[n.buses.carrier == "co2 stored"].x = co2_stored_x.values
    n.buses[n.buses.carrier == "co2 stored"].y = co2_stored_y.values
    """

    n.madd(
        "Link",
        spatial.co2.vents,
        bus0=spatial.co2.nodes,
        bus1="co2 atmosphere",
        carrier="co2 vent",
        efficiency=1.0,
        p_nom_extendable=True,
    )
    if options["co2_network"]:
        # logger.info("Adding CO2 network.")
        co2_links = create_network_topology(n, "CO2 pipeline ")

        cost_onshore = (
            (1 - co2_links.underwater_fraction)
            * costs.at["CO2 pipeline", "fixed"]
            * co2_links.length
        )
        cost_submarine = (
            co2_links.underwater_fraction
            * costs.at["CO2 submarine pipeline", "fixed"]
            * co2_links.length
        )
        capital_cost = cost_onshore + cost_submarine

        n.madd(
            "Link",
            co2_links.index,
            bus0=co2_links.bus0.values + " co2 stored",
            bus1=co2_links.bus1.values + " co2 stored",
            p_min_pu=-1,
            p_nom_extendable=True,
            length=co2_links.length.values,
            capital_cost=capital_cost.values,
            carrier="CO2 pipeline",
            lifetime=costs.at["CO2 pipeline", "lifetime"],
        )

        n.madd(
            "Store",
            spatial.co2.nodes,
            e_nom_extendable=True,
            e_nom_max=np.inf,
            capital_cost=options["co2_sequestration_cost"],
            carrier="co2 stored",
            bus=spatial.co2.nodes,
        )

        # logger.info("Adding CO2 network.")
        co2_links = create_network_topology(n, "CO2 pipeline ")

        cost_onshore = (
            (1 - co2_links.underwater_fraction)
            * costs.at["CO2 pipeline", "fixed"]
            * co2_links.length
        )
        cost_submarine = (
            co2_links.underwater_fraction
            * costs.at["CO2 submarine pipeline", "fixed"]
            * co2_links.length
        )
        capital_cost = cost_onshore + cost_submarine
    cap = co2_cap_from_config(snakemake.config, investment_year)
    n.global_constraints.at["CO2Limit", "constant"] = cap * Nyears
    print(f"[co2] CO2 cap set to {cap:.4g} MWh-equivalent per year (total {cap*Nyears:.4g} over {Nyears} years)")
    print(n.global_constraints.at["CO2Limit", "constant"])

def rescale_to_mapping(p_set_series, mapping):

    # current implied annual energy
    current_energy = float(p_set_series.sum().sum()) 

    factor = mapping / current_energy
    scaled = p_set_series * factor

    print(
        f"rescale factor={factor:.4g}, "
        f"current={current_energy:.4g} MWh/a → target={mapping:.4g} MWh/a"
    )
    return factor

def _assert_no_nans_in_timeseries(n):
    import numpy as np

    # loads
    if hasattr(n, "loads_t") and hasattr(n.loads_t, "p_set") and not n.loads_t.p_set.empty:
        if not np.isfinite(n.loads_t.p_set.to_numpy()).all():
            bad = ~np.isfinite(n.loads_t.p_set.to_numpy())
            raise ValueError(f"NaN/inf in loads_t.p_set (count={bad.sum()})")

    # generators availability
    if hasattr(n, "generators_t") and hasattr(n.generators_t, "p_max_pu") and not n.generators_t.p_max_pu.empty:
        if not np.isfinite(n.generators_t.p_max_pu.to_numpy()).all():
            bad = ~np.isfinite(n.generators_t.p_max_pu.to_numpy())
            raise ValueError(f"NaN/inf in generators_t.p_max_pu (count={bad.sum()})")

    # links availability (if present in your version)
    if hasattr(n, "links_t") and hasattr(n.links_t, "p_max_pu") and not n.links_t.p_max_pu.empty:
        if not np.isfinite(n.links_t.p_max_pu.to_numpy()).all():
            bad = ~np.isfinite(n.links_t.p_max_pu.to_numpy())
            raise ValueError(f"NaN/inf in links_t.p_max_pu (count={bad.sum()})")


def _assert_component_bounds_sane(n):
    def check(df, name):
        if df.empty:
            return

        # ---- 1) NaNs are never allowed (inf is fine = unbounded)
        for col in ["p_nom_min", "p_nom_max", "p_nom"]:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                if s.isna().any():
                    bad = df.loc[s.isna()]
                    raise ValueError(
                        f"NaN in {name}.{col}\n"
                        f"{bad.head(20)}"
                    )

        has_bounds = {"p_nom_min", "p_nom_max", "p_nom_extendable", "p_nom"}.issubset(df.columns)

        if not has_bounds:
            return

        ext = df.p_nom_extendable.fillna(False).astype(bool)

        pmin = pd.to_numeric(df.p_nom_min, errors="coerce").fillna(0.0)
        pmax = pd.to_numeric(df.p_nom_max, errors="coerce")
        pnom = pd.to_numeric(df.p_nom, errors="coerce").fillna(0.0)

        # ---- 2) extendable: p_nom_max must not fall below p_nom_min (finite case)
        bad_ext = ext & np.isfinite(pmax) & (pmax < pmin - 1e-9)

        if bad_ext.any():
            raise ValueError(
                f"{name}: extendable assets with p_nom_max < p_nom_min\n"
                f"{df.loc[bad_ext].head(20)}"
            )

        # ---- 3) extendable: p_nom must respect finite p_nom_max
        bad_ext_nom = ext & np.isfinite(pmax) & (pnom > pmax + 1e-9)

        if bad_ext_nom.any():
            raise ValueError(
                f"{name}: extendable assets with p_nom > p_nom_max\n"
                f"{df.loc[bad_ext_nom].head(20)}"
            )

        # ---- 4) fixed assets: p_nom must not be below p_nom_min  (THIS CAUGHT YOUR 2040 BUG)
        bad_fix = (~ext) & (pnom < pmin - 1e-9)

        if bad_fix.any():
            raise ValueError(
                f"{name}: fixed assets with p_nom < p_nom_min\n"
                f"{df.loc[bad_fix].head(20)}"
            )

    check(n.generators, "generators")
    check(n.links, "links")
    check(n.stores, "stores")
    check(n.storage_units, "storage_units")



def add_aviation(n, costs):
    """
    PATCHES APPLIED (oil/aviation):
    1) reindex airport p_set to spatial.nodes and fill NaN with 0.0 BEFORE adding loads
       (your concat-with-ind created NaNs like TH31 0)
    2) keep carrier names unchanged; only fix p_set creation robustness.
    3) CO2 domestic share block kept; iso2 extraction unchanged.
    """
    sec = options

    corr_aviation = bool(sec.get("correction_aviation", False))
    ASEAN_corr = float(sec.get("correction_ASEAN", 0.0))
    
    dom_av = energy_totals["total domestic aviation"].reindex(countries)
    intl_av = energy_totals["total international aviation"].reindex(countries)
    
    if snakemake.config["sector"]["international_bunkers"]:
        if corr_aviation:
            intl_av=intl_av.fillna(0.0)*ASEAN_corr
        aviation_demand = float((dom_av.fillna(0.0) + intl_av.fillna(0.0)).sum())
        print("int bunkers included in aviation demand calculation")
    else:
        aviation_demand = float(dom_av.sum())

    print("[aviation] total aviation demand [TWh/a]:", aviation_demand)
    airports = pd.read_csv(snakemake.input.airports, keep_default_na=False)
    airports = airports[airports.country.isin(countries)]

    gadm_layer_id = snakemake.config["build_shape_options"]["gadm_layer_id"]

    airports = locate_bus(
        airports,
        countries,
        gadm_layer_id,
        snakemake.input.shapes_path,
        snakemake.config["cluster_options"]["alternative_clustering"],
    ).set_index(f"gadm_{gadm_layer_id}")

    airports["fraction"] = airports["fraction"] / airports["fraction"].sum()
    airports["p_set"] = airports["fraction"] * aviation_demand * 1e6 / 8760

    # --- PATCH: align to spatial.nodes, fill missing with 0 ---
    ind = pd.Index(n.buses.index[n.buses.carrier == "AC"]).astype(str)
    airports = airports.groupby(airports.index).sum(numeric_only=True)
    airports = airports.reindex(ind).copy()
    airports["p_set"] = airports["p_set"].fillna(0.0)

    n.madd(
        "Load",
        spatial.nodes,
        suffix=" kerosene for aviation",
        bus=spatial.oil.nodes,
        carrier="kerosene for aviation",
        p_set=airports["p_set"].reindex(pd.Index(spatial.nodes).astype(str)).fillna(0.0).values,
    )

    
    co2 = float(airports["p_set"].sum()) * costs.at["oil", "CO2 intensity"]
    n.add(
        "Load",
        "aviation oil emissions",
        bus="co2 atmosphere",
        carrier="oil emissions",
        p_set=-co2,
    )
    print("[aviation] CO2 emissions from domestic aviation:", co2)

def add_storage(n, costs):
    "function to add the different types of storage systems"
    logger.info("Add battery storage")

    n.add("Carrier", "battery")

    n.madd(
        "Bus",
        spatial.nodes + " battery",
        location=spatial.nodes,
        carrier="battery",
        x=n.buses.loc[list(spatial.nodes)].x.values,
        y=n.buses.loc[list(spatial.nodes)].y.values,
    )

    n.madd(
        "Store",
        spatial.nodes + " battery",
        bus=spatial.nodes + " battery",
        e_cyclic=True,
        e_nom_extendable=True,
        carrier="battery",
        capital_cost=0,#costs.at["battery storage", "fixed"],
        lifetime=costs.at["battery storage", "lifetime"],
    )
    print(f"Battery costs= {costs.at['battery storage', 'fixed']}")
    print(f"Inverter costs= {costs.at['battery inverter', 'fixed']}")
    
    n.madd(
        "Link",
        spatial.nodes + " battery charger",
        bus0=spatial.nodes,
        bus1=spatial.nodes + " battery",
        carrier="battery charger",
        efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
        capital_cost=costs.at["battery inverter", "fixed"],
        p_nom_extendable=True,
        lifetime=costs.at["battery inverter", "lifetime"],
    )

    n.madd(
        "Link",
        spatial.nodes + " battery discharger",
        bus0=spatial.nodes + " battery",
        bus1=spatial.nodes,
        carrier="battery discharger",
        efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
        marginal_cost=options["marginal_cost_storage"],
        p_nom_extendable=True,
        lifetime=costs.at["battery inverter", "lifetime"],
    )


def h2_hc_conversions(n, costs):
    "function to add the conversion technologies between H2 and hydrocarbons"
    if options["methanation"]:
        n.madd(
            "Link",
            spatial.nodes,
            suffix=" Sabatier",
            bus0=spatial.nodes + " H2",
            bus1=spatial.gas.nodes,
            bus2=spatial.co2.nodes,
            p_nom_extendable=True,
            carrier="Sabatier",
            efficiency=costs.at["methanation", "efficiency"],
            efficiency2=-costs.at["methanation", "efficiency"]
            * costs.at["gas", "CO2 intensity"],
            # costs given per kW_gas
            capital_cost=costs.at["methanation", "fixed"]
            * costs.at["methanation", "efficiency"],
            lifetime=costs.at["methanation", "lifetime"],
        )

    if options["helmeth"]:
        n.madd(
            "Link",
            spatial.nodes,
            suffix=" helmeth",
            bus0=spatial.nodes,
            bus1=spatial.gas.nodes,
            bus2=spatial.co2.nodes,
            carrier="helmeth",
            p_nom_extendable=True,
            efficiency=costs.at["helmeth", "efficiency"],
            efficiency2=-costs.at["helmeth", "efficiency"]
            * costs.at["gas", "CO2 intensity"],
            capital_cost=costs.at["helmeth", "fixed"],
            lifetime=costs.at["helmeth", "lifetime"],
        )

    if options["SMR CC"]:
        if snakemake.config["sector"]["hydrogen"]["hydrogen_colors"]:
            n.madd(
                "Bus",
                spatial.nodes + " blue H2",
                location=spatial.nodes,
                carrier="blue H2",
                x=n.buses.loc[list(spatial.nodes)].x.values,
                y=n.buses.loc[list(spatial.nodes)].y.values,
            )

            n.madd(
                "Link",
                spatial.nodes,
                suffix=" SMR CC",
                bus0=spatial.gas.nodes,
                bus1=spatial.nodes + " blue H2",
                bus2="co2 atmosphere",
                bus3=spatial.co2.nodes,
                p_nom_extendable=True,
                carrier="SMR CC",
                efficiency=costs.at["SMR CC", "efficiency"],
                efficiency2=costs.at["gas", "CO2 intensity"]
                * (1 - options["cc_fraction"]),
                efficiency3=costs.at["gas", "CO2 intensity"] * options["cc_fraction"],
                capital_cost=costs.at["SMR CC", "fixed"],
                lifetime=costs.at["SMR CC", "lifetime"],
            )

            n.madd(
                "Link",
                spatial.nodes + " blue H2",
                bus0=spatial.nodes + " blue H2",
                bus1=spatial.nodes + " H2",
                carrier="blue H2",
                capital_cost=0,
                p_nom_extendable=True,
                # lifetime=costs.at["battery inverter", "lifetime"],
            )

        else:
            n.madd(
                "Link",
                spatial.nodes,
                suffix=" SMR CC",
                bus0=spatial.gas.nodes,
                bus1=spatial.nodes + " H2",
                bus2="co2 atmosphere",
                bus3=spatial.co2.nodes,
                p_nom_extendable=True,
                carrier="SMR CC",
                efficiency=costs.at["SMR CC", "efficiency"],
                efficiency2=costs.at["gas", "CO2 intensity"]
                * (1 - options["cc_fraction"]),
                efficiency3=costs.at["gas", "CO2 intensity"] * options["cc_fraction"],
                capital_cost=costs.at["SMR CC", "fixed"],
                lifetime=costs.at["SMR CC", "lifetime"],
            )

    if options["SMR"]:
        if snakemake.config["sector"]["hydrogen"]["hydrogen_colors"]:
            n.madd(
                "Bus",
                spatial.nodes + " grey H2",
                location=spatial.nodes,
                carrier="grey H2",
                x=n.buses.loc[list(spatial.nodes)].x.values,
                y=n.buses.loc[list(spatial.nodes)].y.values,
            )

            n.madd(
                "Link",
                spatial.nodes + " SMR",
                bus0=spatial.gas.nodes,
                bus1=spatial.nodes + " grey H2",
                bus2="co2 atmosphere",
                p_nom_extendable=True,
                carrier="SMR",
                efficiency=costs.at["SMR", "efficiency"],
                efficiency2=costs.at["gas", "CO2 intensity"],
                capital_cost=costs.at["SMR", "fixed"],
                lifetime=costs.at["SMR", "lifetime"],
            )

            n.madd(
                "Link",
                spatial.nodes + " grey H2",
                bus0=spatial.nodes + " grey H2",
                bus1=spatial.nodes + " H2",
                carrier="grey H2",
                capital_cost=0,
                p_nom_extendable=True,
                # lifetime=costs.at["battery inverter", "lifetime"],
            )

        else:
            n.madd(
                "Link",
                spatial.nodes + " SMR",
                bus0=spatial.gas.nodes,
                bus1=spatial.nodes + " H2",
                bus2="co2 atmosphere",
                p_nom_extendable=True,
                carrier="SMR",
                efficiency=costs.at["SMR", "efficiency"],
                efficiency2=costs.at["gas", "CO2 intensity"],
                capital_cost=costs.at["SMR", "fixed"],
                lifetime=costs.at["SMR", "lifetime"],
            )


def add_shipping(n, costs):
    """
    PATCHES APPLIED (oil/shipping):
    1) same p_set NaN fix as aviation: reindex to all AC nodes and fill NaN with 0.
    2) keep oil store/generator creation but move them BEFORE loads (ensures buses exist).
    3) fix domestic share bug in CO2 block (your code accidentally used aviation series once).
    """

    ports = pd.read_csv(snakemake.input.ports, index_col=None, keep_default_na=False).squeeze()
    ports = ports[ports.country.isin(countries)]

    gadm_layer_id = snakemake.config["build_shape_options"]["gadm_layer_id"]

    sec = options

    corr_shipping = bool(sec.get("correction_shipping", False))
    print(corr_shipping)
    correction_ASEAN = float(sec.get("correction_ASEAN", 0.0))
  
    dom_nav = energy_totals["total domestic navigation"].reindex(countries)
    intl_nav = energy_totals["total international navigation"].reindex(countries)
    print(intl_nav.index)
    

    
    if snakemake.config["sector"]["international_bunkers"]:

        if corr_shipping:
            intl_nav= intl_nav * correction_ASEAN
        
        navigation_demand = float((dom_nav + intl_nav).sum())
        print("int bunkers included in shipping demand calculation")

    else:
        navigation_demand = float(dom_nav.sum())

    print("[shipping] navigation demand [TWh/a]:", navigation_demand)

    efficiency = options["shipping_average_efficiency"] / costs.at["fuel cell", "efficiency"]

    shipping_hydrogen_share = get(
        options["shipping_hydrogen_share"], demand_sc + "_" + str(investment_year)
    )

    ports = locate_bus(
        ports,
        countries,
        gadm_layer_id,
        snakemake.input.shapes_path,
        snakemake.config["cluster_options"]["alternative_clustering"],
    ).set_index(f"gadm_{gadm_layer_id}")

    ports["fraction"] = ports["fraction"] / ports["fraction"].sum()
    ports["p_set"] = shipping_hydrogen_share * ports["fraction"] * navigation_demand * efficiency * 1e6 / 8760

    # --- PATCH: align to all AC nodes, fill NaN p_set with 0 ---
    ind = pd.Index(n.buses.index[n.buses.carrier == "AC"]).astype(str)
    ports = ports.groupby(ports.index).sum(numeric_only=True)
    ports = ports.reindex(ind).copy()
    ports["p_set"] = ports["p_set"].fillna(0.0)

    # liquefaction block unchanged
    if options.get("shipping_hydrogen_liquefaction", False):
        n.madd("Bus", spatial.nodes, suffix=" H2 liquid", carrier="H2 liquid", location=spatial.nodes)
        n.madd(
            "Link",
            spatial.nodes + " H2 liquefaction",
            bus0=spatial.nodes + " H2",
            bus1=spatial.nodes + " H2 liquid",
            carrier="H2 liquefaction",
            efficiency=costs.at["H2 liquefaction", "efficiency"],
            capital_cost=costs.at["H2 liquefaction", "fixed"],
            p_nom_extendable=True,
            lifetime=costs.at["H2 liquefaction", "lifetime"],
        )
        shipping_bus = spatial.nodes + " H2 liquid"
    else:
        shipping_bus = spatial.nodes + " H2"

    if not (
        snakemake.config["policy_config"]["hydrogen"]["is_reference"]
        and snakemake.config["policy_config"]["hydrogen"]["remove_h2_load"]
    ):
        n.madd(
            "Load",
            spatial.nodes,
            suffix=" H2 for shipping",
            bus=shipping_bus,
            carrier="H2 for shipping",
            p_set=ports["p_set"].reindex(pd.Index(spatial.nodes).astype(str)).fillna(0.0).values,
        )

    if shipping_hydrogen_share < 1:
        shipping_oil_share = 1 - shipping_hydrogen_share

        # recompute oil p_set, keep same NaN-proof alignment
        ports_oil = ports.copy()
        ports_oil["p_set"] = shipping_oil_share * ports_oil["fraction"].fillna(0.0) * navigation_demand * 1e6 / 8760
        ports_oil["p_set"] = ports_oil["p_set"].fillna(0.0)

        n.madd(
            "Load",
            spatial.nodes,
            suffix=" shipping oil",
            bus=spatial.oil.nodes,
            carrier="shipping oil",
            p_set=ports_oil["p_set"].reindex(pd.Index(spatial.nodes).astype(str)).fillna(0.0).values,
        )
        
        co2 = float(ports_oil["p_set"].sum()) * costs.at["oil", "CO2 intensity"]
        n.add("Load",
            "shipping oil emissions",
            bus="co2 atmosphere",
            carrier="shipping oil emissions",
            p_set=-co2,
        )
        
def add_industry(n, costs):
    """
    PATCHES APPLIED (industry / biomass-for-industry + general robustness)
    - FIX 1: biomass-for-industry p_set must be aligned to *spatial.biomass.industry* (index match)
            and must never become a giant scalar-on-every-bus by accident.
    - FIX 2: ensure required buses exist (industry biomass bus, biomass fuel bus, CO2 buses).
    - FIX 3: avoid undefined `nodes` variable (use spatial.nodes everywhere).
    - FIX 4: make p_set numeric + NaN-safe.
    """

    logger.info("adding industrial demand")

    industrial_demand = pd.read_csv(snakemake.input.industrial_demand, index_col=0, header=0)
    industrial_demand = industrial_demand.copy()
    # --- scale industry demand by country/year (mapping in MWh/a) ---
    _mapping = load.get("industry", {}).get(investment_year, {})
    if _mapping:
        industrial_demand_sum = industrial_demand.drop(columns=["process emissions"], errors="ignore")
        industrial_demand = industrial_demand * rescale_to_mapping(industrial_demand_sum, _mapping)
    
    process_emissions_correction = load.get("process_emissions_correction", {})
    if process_emissions_correction: 
        industrial_demand = industrial_demand * process_emissions_correction

    # -----------------------------
    # SOLID BIOMASS FOR INDUSTRY
    # -----------------------------
    # Ensure carrier exists
    if "solid biomass for industry" not in n.carriers.index:
        n.add("Carrier", "solid biomass for industry")

    # Ensure upstream fuel carrier exists too (used by bus0 in the link)
    if "solid biomass" not in n.carriers.index:
        n.add("Carrier", "solid biomass")

    # Ensure biomass fuel buses exist (your biomass section may create these; this is a safety net)
    missing_fuel_buses = pd.Index(spatial.biomass.nodes).difference(n.buses.index)
    if len(missing_fuel_buses):
        # best-effort coords via matching AC node name
        base = missing_fuel_buses.str.replace(r" solid biomass$", "", regex=True)
        n.madd(
            "Bus",
            missing_fuel_buses,
            location=base,
            carrier="solid biomass",
            x=n.buses.reindex(base).x.values,
            y=n.buses.reindex(base).y.values,
        )

    # Industry biomass demand buses (these are the endpoints where the Load sits)
    missing_ind_buses = pd.Index(spatial.biomass.industry).difference(n.buses.index)
    if len(missing_ind_buses):
        n.madd(
            "Bus",
            missing_ind_buses,
            location=spatial.biomass.locations,
            carrier="solid biomass for industry",
        )

    # --- build p_set (MW) ---
    # NOTE: industrial_demand is typically indexed by country/region keys; your previous code
    # was mixing "locations" vs "industry bus names" and could become a scalar.
    #
    # Goal:
    #   if biomass_transport=True -> nodal MW per industry bus (Series indexed by spatial.biomass.industry)
    #   else -> a single scalar MW applied uniformly by PyPSA (ok), BUT do this intentionally.

    # Pull the biomass column safely
    if "solid biomass" not in industrial_demand.columns:
        industrial_demand["solid biomass"] = 0.0

    col = pd.to_numeric(industrial_demand["solid biomass"], errors="coerce").fillna(0.0)

    if options.get("biomass_transport", False):
        # Expect industrial demand to be keyed by "spatial.biomass.locations" (e.g. countries)
        # then map each industry bus to its location.
        # spatial.biomass.industry are bus names; spatial.biomass.locations are their "country/region" tags.
        loc_by_bus = pd.Series(
            data=np.array(spatial.biomass.locations, dtype=object),
            index=pd.Index(spatial.biomass.industry, dtype=str),
        )

        # location -> annual MWh; then divide by 8760 -> MW
        # If loc not found in industrial_demand index, default 0.
        annual_mwh_by_loc = col.reindex(pd.Index(spatial.biomass.locations)).fillna(0.0)

        p_set = loc_by_bus.map(annual_mwh_by_loc).fillna(0.0) / 8760.0
        p_set = p_set.reindex(pd.Index(spatial.biomass.industry, dtype=str)).fillna(0.0)
    else:
        # Global scalar MW
        p_set = float(col.sum()) / 8760.0

    # Add biomass-for-industry load
    n.madd(
        "Load",
        spatial.biomass.industry,
        bus=spatial.biomass.industry,
        carrier="solid biomass for industry",
        p_set=p_set,
    )

    # Link from fuel bus -> industry biomass bus
    # IMPORTANT: bus0 must match the fuel buses you actually created in add_biomass:
    # typically spatial.biomass.nodes are like "<ACnode> solid biomass"
    n.madd(
        "Link",
        spatial.biomass.industry,
        bus0=spatial.biomass.nodes,
        bus1=spatial.biomass.industry,
        carrier="solid biomass for industry",
        p_nom_extendable=True,
        efficiency=1.0,
    )

    # Optional CC for industry biomass
    if snakemake.config["sector"].get("cc", False):
        # Ensure CO2 infrastructure exists
        if "co2 atmosphere" not in n.buses.index:
            n.add("Bus", "co2 atmosphere", location="Earth", carrier="co2")
        missing_co2_nodes = pd.Index(spatial.co2.nodes).difference(n.buses.index)
        if len(missing_co2_nodes):
            n.madd("Bus", missing_co2_nodes, location=spatial.co2.locations, carrier="co2 stored")

        n.madd(
            "Link",
            spatial.biomass.industry_cc,
            bus0=spatial.biomass.nodes,
            bus1=spatial.biomass.industry,
            bus2="co2 atmosphere",
            bus3=spatial.co2.nodes,
            carrier="solid biomass for industry CC",
            p_nom_extendable=True,
            capital_cost=costs.at["cement capture", "fixed"] * costs.at["solid biomass", "CO2 intensity"],
            efficiency=0.9,  # TODO config
            efficiency2=-costs.at["solid biomass", "CO2 intensity"] * costs.at["cement capture", "capture_rate"],
            efficiency3= costs.at["solid biomass", "CO2 intensity"] * costs.at["cement capture", "capture_rate"],
            lifetime=costs.at["cement capture", "lifetime"],
        )

    # -----------------------------
    # GAS FOR INDUSTRY
    # -----------------------------
    if "gas for industry" not in n.carriers.index:
        n.add("Carrier", "gas for industry")

    missing_gas_ind_buses = pd.Index(spatial.gas.industry).difference(n.buses.index)
    if len(missing_gas_ind_buses):
        n.madd("Bus", missing_gas_ind_buses, location=spatial.gas.locations, carrier="gas for industry")

    if "gas" not in industrial_demand.columns:
        industrial_demand["gas"] = 0.0

    gas_demand = pd.to_numeric(industrial_demand.loc[spatial.nodes, "gas"], errors="coerce").fillna(0.0) / 8760.0

    if options.get("gas", {}).get("spatial_gas", False):
        spatial_gas_demand = gas_demand.rename(index=lambda x: x + " gas for industry")
    else:
        spatial_gas_demand = float(gas_demand.sum())

    n.madd(
        "Load",
        spatial.gas.industry,
        bus=spatial.gas.industry,
        carrier="gas for industry",
        p_set=spatial_gas_demand,
    )

    n.madd(
        "Link",
        spatial.gas.industry,
        bus0=spatial.gas.nodes,
        bus1=spatial.gas.industry,
        bus2="co2 atmosphere",
        carrier="gas for industry",
        p_nom_extendable=True,
        efficiency=1.0,
        efficiency2=costs.at["gas", "CO2 intensity"],
    )

    if snakemake.config["sector"].get("cc", False):
        n.madd(
            "Link",
            spatial.gas.industry_cc,
            bus0=spatial.gas.nodes,
            bus1=spatial.gas.industry,
            bus2="co2 atmosphere",
            bus3=spatial.co2.nodes,
            carrier="gas for industry CC",
            p_nom_extendable=True,
            capital_cost=costs.at["cement capture", "fixed"] * costs.at["gas", "CO2 intensity"],
            efficiency=0.9,
            efficiency2=costs.at["gas", "CO2 intensity"] * (1 - costs.at["cement capture", "capture_rate"]),
            efficiency3=costs.at["gas", "CO2 intensity"] * costs.at["cement capture", "capture_rate"],
            lifetime=costs.at["cement capture", "lifetime"],
        )

    # -----------------------------
    # H2 FOR INDUSTRY (fix undefined `nodes`)
    # -----------------------------
    if "hydrogen" not in industrial_demand.columns:
        industrial_demand["hydrogen"] = 0.0

    if not (
        snakemake.config["policy_config"]["hydrogen"]["is_reference"]
        and snakemake.config["policy_config"]["hydrogen"]["remove_h2_load"]
    ):
        n.madd(
            "Load",
            spatial.nodes,
            suffix=" H2 for industry",
            bus=spatial.nodes + " H2",
            carrier="H2 for industry",
            p_set=pd.to_numeric(industrial_demand.loc[spatial.nodes, "hydrogen"], errors="coerce").fillna(0.0) / 8760.0,
        )

    # -----------------------------
    # OIL / NAPHTHA FOR INDUSTRY + emissions
    # -----------------------------
    if "oil" not in industrial_demand.columns:
        industrial_demand["oil"] = 0.0

    n.madd(
        "Load",
        spatial.nodes,
        suffix=" naphtha for industry",
        bus=spatial.oil.nodes,
        carrier="naphtha for industry",
        p_set=pd.to_numeric(industrial_demand.loc[spatial.nodes, "oil"], errors="coerce").fillna(0.0) / 8760.0,
    )

    co2_oil = (
        n.loads.loc[spatial.nodes + " naphtha for industry", "p_set"].sum()
        * costs.at["oil", "CO2 intensity"]
    )
    n.add("Load", "industry oil emissions", bus="co2 atmosphere", carrier="industry oil emissions", p_set=-float(co2_oil))

    # COAL 
    # ensure coal carrier/buses exist in your system somewhere (like you do for oil/gas)

    # add coal fuel consumption load (spatial)
    if "coal" not in industrial_demand.columns:
        industrial_demand["coal"] = 0.0

    coal_p_set = pd.to_numeric(
        industrial_demand.loc[spatial.nodes, "coal"], errors="coerce"
    ).fillna(0.0) / 8760.0

    n.madd(
        "Load",
        spatial.nodes,
        suffix=" coal for industry",
        bus=spatial.coal.nodes,          # you need this analogous to spatial.oil.nodes
        carrier="coal for industry",
        p_set=coal_p_set,
    )

    # add emissions derived from that load (consistent)
    co2_coal = (
        n.loads.loc[spatial.nodes + " coal for industry", "p_set"].sum()
        * costs.at["coal", "CO2 intensity"]
    )
    n.add(
        "Load",
        "industry coal emissions",
        bus="co2 atmosphere",
        carrier="industry coal emissions",
        p_set=-float(co2_coal),
    )

    # -----------------------------
    # LOW-T HEAT FOR INDUSTRY
    # -----------------------------
    if "low-temperature heat" not in industrial_demand.columns:
        industrial_demand["low-temperature heat"] = 0.0

    n.madd(
        "Load",
        spatial.nodes,
        suffix=" low-temperature heat for industry",
        bus=[
            (node + " urban central heat") if (node + " urban central heat") in n.buses.index
            else (node + " services urban decentral heat")
            for node in spatial.nodes
        ],
        carrier="low-temperature heat for industry",
        p_set=pd.to_numeric(industrial_demand.loc[spatial.nodes, "low-temperature heat"], errors="coerce").fillna(0.0) / 8760.0,
    )

    # -----------------------------
    # ELECTRICITY FOR INDUSTRY
    # -----------------------------
    if "electricity" not in industrial_demand.columns:
        industrial_demand["electricity"] = 0.0

    n.madd(
        "Load",
        spatial.nodes,
        suffix=" industry electricity",
        bus=spatial.nodes,
        carrier="industry electricity",
        p_set=pd.to_numeric(industrial_demand.loc[spatial.nodes, "electricity"], errors="coerce").fillna(0.0) / 8760.0,
    )

    # -----------------------------
    # PROCESS EMISSIONS (bus + link)
    # -----------------------------
    if "process emissions" not in n.buses.index:
        n.add("Bus", "process emissions", location="Earth", carrier="process emissions")

    if "process emissions" not in industrial_demand.columns:
        industrial_demand["process emissions"] = 0.0

    n.madd(
        "Load",
        spatial.nodes,
        suffix=" process emissions",
        bus="process emissions",
        carrier="process emissions",
        p_set=-(pd.to_numeric(industrial_demand.loc[spatial.nodes, "process emissions"], errors="coerce").fillna(0.0) / 8760.0),
    )

    if "process emissions" not in n.links.index:
        n.add(
            "Link",
            "process emissions",
            bus0="process emissions",
            bus1="co2 atmosphere",
            carrier="process emissions",
            p_nom_extendable=True,
            efficiency=1.0,
        )

    if snakemake.config["sector"].get("cc", False):
        n.madd(
            "Link",
            spatial.co2.locations,
            suffix=" process emissions CC",
            bus0="process emissions",
            bus1="co2 atmosphere",
            bus2=spatial.co2.nodes,
            carrier="process emissions CC",
            p_nom_extendable=True,
            capital_cost=costs.at["cement capture", "fixed"],
            efficiency=1 - costs.at["cement capture", "capture_rate"],
            efficiency2=costs.at["cement capture", "capture_rate"],
            lifetime=costs.at["cement capture", "lifetime"],
        )



def get(item, investment_year=None):
    """
    Check whether item depends on investment year.
    """
    if isinstance(item, dict):
        return item[investment_year]
    else:
        return item


"""
Missing data:
 - transport
 - aviation data
 - nodal_transport_data
 - cycling_shift
 - dsm_profile
 - avail_profile
"""


def add_land_transport(n, costs):
    """
    Function to add land transport to network.
    """
    # TODO options?

    logger.info("adding land transport")
    

    if options["dynamic_transport"]["enable"] == False:
        fuel_cell_share = get(
            options["land_transport_fuel_cell_share"],
            demand_sc + "_" + str(investment_year),
        )
        electric_share = get(
            options["land_transport_electric_share"],
            demand_sc + "_" + str(investment_year),
        )

    elif options["dynamic_transport"]["enable"] == True:
        fuel_cell_share = options["dynamic_transport"][
            "land_transport_fuel_cell_share"
        ][snakemake.wildcards.opts]
        electric_share = options["dynamic_transport"]["land_transport_electric_share"][
            snakemake.wildcards.opts
        ]

    ice_share = 1 - fuel_cell_share - electric_share

    logger.info("FCEV share: {}".format(fuel_cell_share))
    logger.info("EV share: {}".format(electric_share))
    logger.info("ICEV share: {}".format(ice_share))

    assert ice_share >= 0, "Error, more FCEV and EV share than 1."

    # Nodes are already defined, remove it from here
    # nodes = pop_layout.index

    if electric_share > 0:
        n.add("Carrier", "Li ion")

        n.madd(
            "Bus",
            spatial.nodes,
            location=spatial.nodes,
            suffix=" EV battery",
            carrier="Li ion",
            x=n.buses.loc[list(spatial.nodes)].x.values,
            y=n.buses.loc[list(spatial.nodes)].y.values,
        )

        p_set = (
            electric_share
            * (
                transport[spatial.nodes]
                + cycling_shift(transport[spatial.nodes], 1)
                + cycling_shift(transport[spatial.nodes], 2)
            )
            / 3 
        )

        n.madd(
            "Load",
            spatial.nodes,
            suffix=" land transport EV",
            bus=spatial.nodes + " EV battery",
            carrier="land transport EV",
            p_set=p_set,
        )

        p_nom = (
            nodal_transport_data["number cars"]
            * options.get("bev_charge_rate", 0.011)
            * electric_share
        )

        n.madd(
            "Link",
            spatial.nodes,
            suffix=" BEV charger",
            bus0=spatial.nodes,
            bus1=spatial.nodes + " EV battery",
            p_nom=p_nom,
            carrier="BEV charger",
            p_max_pu=avail_profile[spatial.nodes],
            efficiency=options.get("bev_charge_efficiency", 0.9),
            # These were set non-zero to find LU infeasibility when availability = 0.25
            # p_nom_extendable=True,
            # p_nom_min=p_nom,
            # capital_cost=1e6,  #i.e. so high it only gets built where necessary
        )

    if electric_share > 0 and options["v2g"]:
        n.madd(
            "Link",
            spatial.nodes,
            suffix=" V2G",
            bus1=spatial.nodes,
            bus0=spatial.nodes + " EV battery",
            p_nom=p_nom,
            carrier="V2G",
            p_max_pu=avail_profile[spatial.nodes],
            efficiency=options.get("bev_charge_efficiency", 0.9),
        )

    if electric_share > 0 and options["bev_dsm"]:
        e_nom = (
            nodal_transport_data["number cars"]
            * options.get("bev_energy", 0.05)
            * options["bev_availability"]
            * electric_share
        )

        n.madd(
            "Store",
            spatial.nodes,
            suffix=" battery storage",
            bus=spatial.nodes + " EV battery",
            carrier="battery storage",
            e_cyclic=True,
            e_nom=e_nom,
            e_max_pu=1,
            e_min_pu=dsm_profile[spatial.nodes],
        )

    if fuel_cell_share > 0:
        if not (
            snakemake.config["policy_config"]["hydrogen"]["is_reference"]
            and snakemake.config["policy_config"]["hydrogen"]["remove_h2_load"]
        ):
            n.madd(
                "Load",
                nodes,
                suffix=" land transport fuel cell",
                bus=nodes + " H2",
                carrier="land transport fuel cell",
                p_set=fuel_cell_share
                / options["transport_fuel_cell_efficiency"]
                * transport[nodes],
            )

    if ice_share > 0:
        if "oil" not in n.buses.carrier.unique():
            n.madd(
                "Bus", spatial.oil.nodes, location=spatial.oil.locations, carrier="oil"
            )
        ice_efficiency = options["transport_internal_combustion_efficiency"]
        print("land transport oil")
        print(ice_share / ice_efficiency * transport[spatial.nodes])

        n.madd(
            "Load",
            spatial.nodes,
            suffix=" land transport oil",
            bus=spatial.oil.nodes,
            carrier="land transport oil",
            p_set=ice_share / ice_efficiency * transport[spatial.nodes],
        )

        co2 = (
            ice_share
            / ice_efficiency
            * transport[spatial.nodes].sum().sum()
            / 8760
            * costs.at["oil", "CO2 intensity"]
        )
        print("CO2 emissions from land transport oil:", co2)
        print('ice share:', ice_share)
        print("ice efficiency:", ice_efficiency)
        print(transport[spatial.nodes].sum().sum())

        n.add(
            "Load",
            "land transport oil emissions",
            bus="co2 atmosphere",
            carrier="land transport oil emissions",
            p_set=-co2,
        )


def create_nodes_for_heat_sector():
    # TODO pop_layout

    # rural are areas with low heating density and individual heating
    # urban are areas with high heating density
    # urban can be split into district heating (central) and individual heating (decentral)

    ct_urban = pop_layout.urban.groupby(pop_layout.ct).sum()
    # distribution of urban population within a country
    pop_layout["urban_ct_fraction"] = pop_layout.urban / pop_layout.ct.map(ct_urban.get)

    sectors = ["residential", "services"]

    h_nodes = {}
    urban_fraction = pop_layout.urban / pop_layout[["rural", "urban"]].sum(axis=1)

    for sector in sectors:
        h_nodes[sector + " rural"] = pop_layout.index
        h_nodes[sector + " urban decentral"] = pop_layout.index

    # maximum potential of urban demand covered by district heating
    central_fraction = options["district_heating"]["potential"]
    # district heating share at each node
    dist_fraction_node = (
        district_heat_share["district heat share"]
        * pop_layout["urban_ct_fraction"]
        / pop_layout["fraction"]
    )
    h_nodes["urban central"] = dist_fraction_node.index
    # if district heating share larger than urban fraction -> set urban
    # fraction to district heating share
    urban_fraction = pd.concat([urban_fraction, dist_fraction_node], axis=1).max(axis=1)
    # difference of max potential and today's share of district heating
    diff = (urban_fraction * central_fraction) - dist_fraction_node
    progress = get(options["district_heating"]["progress"], investment_year)
    dist_fraction_node += diff * progress
    # logger.info(
    #     "The current district heating share compared to the maximum",
    #     f"possible is increased by a progress factor of\n{progress}",
    #     "resulting in a district heating share of",  # "\n{dist_fraction_node}", #TODO fix district heat share
    # )

    return h_nodes, dist_fraction_node, urban_fraction


def add_heat(n, costs):
    # TODO options?
    # TODO pop_layout?

    logger.info("adding heat")
    sectors = ["residential", "services"]

    h_nodes, dist_fraction, urban_fraction = create_nodes_for_heat_sector()

    # NB: must add costs of central heating afterwards (EUR 400 / kWpeak, 50a, 1% FOM from Fraunhofer ISE)

    # exogenously reduce space heat demand
    if options["reduce_space_heat_exogenously"]:
        dE = get(options["reduce_space_heat_exogenously_factor"], investment_year)
        # print(f"assumed space heat reduction of {dE*100} %")
        for sector in sectors:
            heat_demand[sector + " space"] = (1 - dE) * heat_demand[sector + " space"]

    heat_systems = [
        "residential rural",
        "services rural",
        "residential urban decentral",
        "services urban decentral",
        "urban central",
    ]

    for name in heat_systems:
        name_type = "central" if name == "urban central" else "decentral"

        n.add("Carrier", name + " heat")

        n.madd(
            "Bus",
            h_nodes[name] + " {} heat".format(name),
            location=h_nodes[name],
            carrier=name + " heat",
        )

        ## Add heat load

        for sector in sectors:
            rural = 1 - urban_fraction[h_nodes[name]]
            central = dist_fraction[h_nodes[name]]
            decentral = urban_fraction[h_nodes[name]] - dist_fraction[h_nodes[name]]
            global_total = (rural.sum() + central.sum() + decentral.sum())
            
            # heat demand weighting
            if "rural" in name:
                factor = rural / global_total
                print("rural",factor.sum())
            elif "urban central" in name:
                factor = central / global_total
                print("central",factor.sum())
            elif "urban decentral" in name:
                factor = decentral / global_total
                print("decentral",factor.sum())
            else:
                raise NotImplementedError(
                    f" {name} not in " f"heat systems: {heat_systems}"
                )
            
            if sector in name:
                heat_load = (
                    heat_demand[[sector + " water", sector + " space"]]
                    .groupby(level=1, axis=1)
                    .sum()[h_nodes[name]]
                    .multiply(factor)
                )
        if name == "urban central":
            heat_load = (
                heat_demand.groupby(level=1, axis=1)
                .sum()[h_nodes[name]]
                .multiply(
                    factor * (1 + options["district_heating"]["district_heating_loss"])
                )
            )
        
        
        peak = float(heat_load.max().max())  # MW
        


        n.madd(
            "Load",
            h_nodes[name],
            suffix=f" {name} heat",
            bus=h_nodes[name] + f" {name} heat",
            carrier=name + " heat",
            p_set=heat_load,
        )

        ## Add heat pumps

        heat_pump_type = "air" if "urban" in name else "ground"

        costs_name = f"{name_type} {heat_pump_type}-sourced heat pump"
        cop = {"air": ashp_cop, "ground": gshp_cop}
        efficiency = (
            cop[heat_pump_type][h_nodes[name]]
            if options["time_dep_hp_cop"]
            else costs.at[costs_name, "efficiency"]
        )

        n.madd(
            "Link",
            h_nodes[name],
            suffix=f" {name} {heat_pump_type} heat pump",
            bus0=h_nodes[name],
            bus1=h_nodes[name] + f" {name} heat",
            carrier=f"{name} {heat_pump_type} heat pump",
            efficiency=efficiency,
            capital_cost=costs.at[costs_name, "efficiency"]
            * costs.at[costs_name, "fixed"],
            p_nom_extendable=True,
            lifetime=costs.at[costs_name, "lifetime"],
        )

        if options["tes"]:
            n.add("Carrier", name + " water tanks")

            n.madd(
                "Bus",
                h_nodes[name] + f" {name} water tanks",
                location=h_nodes[name],
                carrier=name + " water tanks",
            )

            n.madd(
                "Link",
                h_nodes[name] + f" {name} water tanks charger",
                bus0=h_nodes[name] + f" {name} heat",
                bus1=h_nodes[name] + f" {name} water tanks",
                efficiency=costs.at["water tank charger", "efficiency"],
                carrier=name + " water tanks charger",
                #marginal_cost2=1e-6,
                p_nom_extendable=True,
            )

            n.madd(
                "Link",
                h_nodes[name] + f" {name} water tanks discharger",
                bus0=h_nodes[name] + f" {name} water tanks",
                bus1=h_nodes[name] + f" {name} heat",
                carrier=name + " water tanks discharger",
                efficiency=costs.at["water tank discharger", "efficiency"],
                #marginal_cost2=1e-6,
                p_nom_extendable=True,
            )

            if isinstance(options["tes_tau"], dict):
                tes_time_constant_days = options["tes_tau"][name_type]
            else:  # TODO add logger
                # logger.warning("Deprecated: a future version will require you to specify 'tes_tau' ",
                # "for 'decentral' and 'central' separately.")
                tes_time_constant_days = (
                    options["tes_tau"] if name_type == "decentral" else 180.0
                )

            # conversion from EUR/m^3 to EUR/MWh for 40 K diff and 1.17 kWh/m^3/K
            capital_cost = (
                costs.at[name_type + " water tank storage", "fixed"] / 0.00117 / 40
            )

            n.madd(
                "Store",
                h_nodes[name] + f" {name} water tanks",
                bus=h_nodes[name] + f" {name} water tanks",
                e_cyclic=True,
                e_nom_extendable=True,
                carrier=name + " water tanks",
                standing_loss=1 - np.exp(-1 / 24 / tes_time_constant_days),
                capital_cost=capital_cost,
                lifetime=costs.at[name_type + " water tank storage", "lifetime"],
            )

        if options["boilers"]:
            key = f"{name_type} resistive heater"

            n.madd(
                "Link",
                h_nodes[name] + f" {name} resistive heater",
                bus0=h_nodes[name],
                bus1=h_nodes[name] + f" {name} heat",
                carrier=name + " resistive heater",
                efficiency=costs.at[key, "efficiency"],
                capital_cost=costs.at[key, "efficiency"] * costs.at[key, "fixed"],
                p_nom_extendable=True,
                lifetime=costs.at[key, "lifetime"],
            )

            key = f"{name_type} gas boiler"

            n.madd(
                "Link",
                h_nodes[name] + f" {name} gas boiler",
                p_nom_extendable=True,
                bus0=spatial.gas.nodes,
                bus1=h_nodes[name] + f" {name} heat",
                bus2="co2 atmosphere",
                carrier=name + " gas boiler",
                efficiency=costs.at[key, "efficiency"],
                efficiency2=costs.at["gas", "CO2 intensity"],
                capital_cost=costs.at[key, "efficiency"] * costs.at[key, "fixed"],
                lifetime=costs.at[key, "lifetime"],
            )

        if options["solar_thermal"]:
            n.add("Carrier", name + " solar thermal")

            n.madd(
                "Generator",
                h_nodes[name],
                suffix=f" {name} solar thermal collector",
                bus=h_nodes[name] + f" {name} heat",
                carrier=name + " solar thermal",
                p_nom_extendable=True,
                capital_cost=costs.at[name_type + " solar thermal", "fixed"],
                p_max_pu=solar_thermal.reindex(columns=h_nodes[name], fill_value=0.0),
                lifetime=costs.at[name_type + " solar thermal", "lifetime"],
            )

        if options["chp"] and name == "urban central":
            # add gas CHP; biomass CHP is added in biomass section
            n.madd(
                "Link",
                h_nodes[name] + " urban central gas CHP",
                bus0=spatial.gas.nodes,
                bus1=h_nodes[name],
                bus2=h_nodes[name] + " urban central heat",
                bus3="co2 atmosphere",
                carrier="urban central gas CHP",
                p_nom_extendable=True,
                capital_cost=costs.at["central gas CHP", "fixed"]
                * costs.at["central gas CHP", "efficiency"],
                marginal_cost=costs.at["central gas CHP", "VOM"],
                efficiency=costs.at["central gas CHP", "efficiency"],
                efficiency2=costs.at["central gas CHP", "efficiency"]
                / costs.at["central gas CHP", "c_b"],
                efficiency3=costs.at["gas", "CO2 intensity"],
                lifetime=costs.at["central gas CHP", "lifetime"],
            )
            if snakemake.config["sector"]["cc"]:
                n.madd(
                    "Link",
                    h_nodes[name] + " urban central gas CHP CC",
                    # bus0="Earth gas",
                    bus0=spatial.gas.nodes,
                    bus1=h_nodes[name],
                    bus2=h_nodes[name] + " urban central heat",
                    bus3="co2 atmosphere",
                    bus4=spatial.co2.df.loc[h_nodes[name], "nodes"].values,
                    carrier="urban central gas CHP CC",
                    p_nom_extendable=True,
                    capital_cost=costs.at["central gas CHP", "fixed"]
                    * costs.at["central gas CHP", "efficiency"]
                    + costs.at["biomass CHP capture", "fixed"]
                    * costs.at["gas", "CO2 intensity"],
                    marginal_cost=costs.at["central gas CHP", "VOM"],
                    efficiency=costs.at["central gas CHP", "efficiency"]
                    - costs.at["gas", "CO2 intensity"]
                    * (
                        costs.at["biomass CHP capture", "electricity-input"]
                        + costs.at[
                            "biomass CHP capture", "compression-electricity-input"
                        ]
                    ),
                    efficiency2=costs.at["central gas CHP", "efficiency"]
                    / costs.at["central gas CHP", "c_b"]
                    + costs.at["gas", "CO2 intensity"]
                    * (
                        costs.at["biomass CHP capture", "heat-output"]
                        + costs.at["biomass CHP capture", "compression-heat-output"]
                        - costs.at["biomass CHP capture", "heat-input"]
                    ),
                    efficiency3=costs.at["gas", "CO2 intensity"]
                    * (1 - costs.at["biomass CHP capture", "capture_rate"]),
                    efficiency4=costs.at["gas", "CO2 intensity"]
                    * costs.at["biomass CHP capture", "capture_rate"],
                    lifetime=costs.at["central gas CHP", "lifetime"],
                )

        if options["chp"] and options["micro_chp"] and name != "urban central":
            n.madd(
                "Link",
                h_nodes[name] + f" {name} micro gas CHP",
                p_nom_extendable=True,
                # bus0="Earth gas",
                bus0=spatial.gas.nodes,
                bus1=h_nodes[name],
                bus2=h_nodes[name] + f" {name} heat",
                bus3="co2 atmosphere",
                carrier=name + " micro gas CHP",
                efficiency=costs.at["micro CHP", "efficiency"],
                efficiency2=costs.at["micro CHP", "efficiency-heat"],
                efficiency3=costs.at["gas", "CO2 intensity"],
                capital_cost=costs.at["micro CHP", "fixed"],
                lifetime=costs.at["micro CHP", "lifetime"],
            )


def average_every_nhours(n, offset):
    # logger.info(f'Resampling the network to {offset}')
    m = n.copy(with_time=False)

    snapshot_weightings = n.snapshot_weightings.resample(offset.casefold()).sum()
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name + "_t")
        for k, df in c.pnl.items():
            if not df.empty:
                if c.list_name == "stores" and k == "e_max_pu":
                    pnl[k] = df.resample(offset.casefold()).min()
                elif c.list_name == "stores" and k == "e_min_pu":
                    pnl[k] = df.resample(offset.casefold()).max()
                else:
                    pnl[k] = df.resample(offset.casefold()).mean()

    return m


def add_dac(n, costs):
    heat_carriers = ["urban central heat", "services urban decentral heat"]
    heat_buses = n.buses.index[n.buses.carrier.isin(heat_carriers)]
    locations = n.buses.location[heat_buses]

    efficiency2 = -(
        costs.at["direct air capture", "electricity-input"]
        + costs.at["direct air capture", "compression-electricity-input"]
    )
    efficiency3 = -(
        costs.at["direct air capture", "heat-input"]
        - costs.at["direct air capture", "compression-heat-output"]
    )

    n.madd(
        "Link",
        heat_buses.str.replace(" heat", " DAC"),
        bus0="co2 atmosphere",
        bus1=spatial.co2.df.loc[locations, "nodes"].values,
        bus2=locations.values,
        bus3=heat_buses,
        carrier="DAC",
        capital_cost=costs.at["direct air capture", "fixed"],
        efficiency=1.0,
        efficiency2=efficiency2,
        efficiency3=efficiency3,
        p_nom_extendable=True,
        lifetime=costs.at["direct air capture", "lifetime"],
    )


def add_services(n, costs):
    temporal_resolution = n.snapshot_weightings.generators
    buses = spatial.nodes.intersection(n.loads_t.p_set.columns)
    _mapping = load.get("services", {}).get(investment_year, {})
    profile_residential = normalize_by_country(
        n.loads_t.p_set[buses].reindex(columns=spatial.nodes, fill_value=0.0)
    ).fillna(0)

    p_set_elec = p_set_from_scaling(
        "services electricity", profile_residential, energy_totals, temporal_resolution
    )
    p_set_biomass = p_set_from_scaling(
        "services biomass", profile_residential, energy_totals, temporal_resolution
    )
    p_set_oil = p_set_from_scaling(
        "services oil", profile_residential, energy_totals, temporal_resolution
    )
    p_set_gas = p_set_from_scaling(
        "services gas", profile_residential, energy_totals, temporal_resolution
    )
    p_services = p_set_elec + p_set_biomass + p_set_oil + p_set_gas
    p_services=p_services.mul(temporal_resolution, axis=0)
    print('services')
    if _mapping:
        scale_services = rescale_to_mapping(p_services, _mapping) #energy_totals is in Twh
        p_set_elec = p_set_elec*scale_services
        p_set_biomass = p_set_biomass*scale_services
        p_set_oil = p_set_oil*scale_services
        p_set_gas = p_set_gas*scale_services
        print(scale_services)
    else:
        scale_services = 1
    n.madd(
        "Load",
        spatial.nodes,
        suffix=" services electricity",
        bus=spatial.nodes,
        carrier="services electricity",
        p_set=p_set_elec,
    )


    n.madd(
        "Load",
        spatial.nodes,
        suffix=" services biomass",
        bus=spatial.biomass.nodes,
        carrier="services biomass",
        p_set=p_set_biomass,
    )

    # co2 = (
    #     p_set_biomass.sum().sum() * costs.at["solid biomass", "CO2 intensity"]
    # ) / 8760

    # n.add(
    #     "Load",
    #     "services biomass emissions",
    #     bus="co2 atmosphere",
    #     carrier="biomass emissions",
    #     p_set=-co2,
    # )
    

    n.madd(
        "Load",
        spatial.nodes,
        suffix=" services oil",
        bus=spatial.oil.nodes,
        carrier="services oil",
        p_set=p_set_oil*scale_services,
    )

    # TODO check with different snapshot settings
    co2 = p_set_oil.sum(axis=1).mean() * costs.at["oil", "CO2 intensity"]

    n.add(
        "Load",
        "services oil emissions",
        bus="co2 atmosphere",
        carrier="oil emissions",
        p_set=-co2,
    )


    n.madd(
        "Load",
        spatial.nodes,
        suffix=" services gas",
        bus=spatial.gas.nodes,
        carrier="services gas",
        p_set=p_set_gas,
    )

    # TODO check with different snapshot settings
    co2 = p_set_gas.sum(axis=1).mean() * costs.at["gas", "CO2 intensity"]

    n.add(
        "Load",
        "services gas emissions",
        bus="co2 atmosphere",
        carrier="gas emissions",
        p_set=-co2,
    )


def add_agriculture(n, costs):
    _mapping = load.get("agriculture", {}).get(investment_year, {})
    agri_demand_el=nodal_energy_totals.loc[spatial.nodes, "agriculture electricity"]* 1e6/ 8760 
    agri_demand_oil= nodal_energy_totals.loc[spatial.nodes, "agriculture oil"] * 1e6 / 8760
    agri_demand=agri_demand_el + agri_demand_oil
    
    # same indentation level as agri_demand_el/agri_demand_oil
    if _mapping:
        print('agri')
        H_yr = float(n.snapshot_weightings.generators.sum())  # hours represented in the year
        # pass MWh/a to the mapper (MW * hours)
        scale_agri = rescale_to_mapping(agri_demand * H_yr, _mapping)
        agri_demand_el  = agri_demand_el  * scale_agri
        agri_demand_oil = agri_demand_oil * scale_agri
        print(scale_agri)
    else:
        scale_agri = 1.0

    n.madd(
        "Load",
        spatial.nodes,
        suffix=" agriculture electricity",
        bus=spatial.nodes,
        carrier="agriculture electricity",
        p_set=nodal_energy_totals.loc[spatial.nodes, "agriculture electricity"]
        * 1e6
        / 8760,
    )

    n.madd(
        "Load",
        spatial.nodes,
        suffix=" agriculture oil",
        bus=spatial.oil.nodes,
        carrier="agriculture oil",
        p_set=nodal_energy_totals.loc[spatial.nodes, "agriculture oil"] * 1e6 / 8760,
    )
    co2 = (
        nodal_energy_totals.loc[spatial.nodes, "agriculture oil"]
        * 1e6
        / 8760
        * costs.at["oil", "CO2 intensity"]
    ).sum()

    n.add(
        "Load",
        "agriculture oil emissions",
        bus="co2 atmosphere",
        carrier="oil emissions",
        p_set=-co2,
    )


def normalize_by_country(df, droplevel=False):
    """
    Auxiliary function to normalize a dataframe by the country.

    If droplevel is False (default), the country level is added to the
    column index If droplevel is True, the original column format is
    preserved
    """
    ret = df.T.groupby(df.columns.str[:2]).apply(lambda x: x / x.sum().sum()).T
    if droplevel:
        return ret.droplevel(0, axis=1)
    else:
        return ret


def group_by_node(df, multiindex=False):
    """
    Auxiliary function to group a dataframe by the node name.
    """
    ret = df.T.groupby(df.columns.str.split(" ").str[0]).sum().T
    if multiindex:
        ret.columns = pd.MultiIndex.from_tuples(zip(ret.columns.str[:2], ret.columns))
    return ret


def normalize_and_group(df, multiindex=False):
    """
    Function to concatenate normalize_by_country and group_by_node.
    """
    return group_by_node(
        normalize_by_country(df, droplevel=True), multiindex=multiindex
    )


def p_set_from_scaling(col, scaling, energy_totals, nhours):
    """
    Function to create p_set from energy_totals, using the per-unit scaling
    dataframe.
    """
    return 1e6 * scaling.div(nhours, axis=0).mul(energy_totals[col], level=0).droplevel(
        level=0, axis=1
    )


def add_residential(n, costs):
    # need to adapt for many countries #TODO

    # if snakemake.config["custom_data"]["heat_demand"]:
    # heat_demand_index=n.loads_t.p.filter(like='residential').filter(like='heat').dropna(axis=1).index
    # oil_res_index=n.loads_t.p.filter(like='residential').filter(like='oil').dropna(axis=1).index

    temporal_resolution = n.snapshot_weightings.generators
    _mapping = load.get("residential", {}).get(investment_year, {})
    heat_ind = (
        n.loads_t.p_set.filter(like="residential")
        .filter(like="heat")
        .dropna(axis=1)
        .columns
    )
    heat_shape_raw = normalize_by_country(n.loads_t.p_set[heat_ind])
    heat_shape = heat_shape_raw.rename(
        columns=n.loads.bus.map(n.buses.location), level=1
    )
    heat_shape = heat_shape.T.groupby(level=[0, 1]).sum().T

    n.loads_t.p_set[heat_ind] = 1e6 * heat_shape_raw.mul(
        energy_totals["total residential space"]
        + energy_totals["total residential water"]
        - energy_totals["residential heat biomass"]
        - energy_totals["residential heat oil"]
        - energy_totals["residential heat gas"],
        level=0,
    ).droplevel(level=0, axis=1).div(temporal_resolution, axis=0)

    heat_oil_demand = p_set_from_scaling(
        "residential heat oil", heat_shape, energy_totals, temporal_resolution
    )
    heat_biomass_demand = p_set_from_scaling(
        "residential heat biomass", heat_shape, energy_totals, temporal_resolution
    )

    heat_gas_demand = p_set_from_scaling(
        "residential heat gas", heat_shape, energy_totals, temporal_resolution
    )

    res_index = spatial.nodes.intersection(n.loads_t.p_set.columns)
    profile_residential_raw = normalize_by_country(n.loads_t.p_set[res_index])
    profile_residential = profile_residential_raw.rename(
        columns=n.loads.bus.map(n.buses.location), level=1
    )
    profile_residential = profile_residential.T.groupby(level=[0, 1]).sum().T
    p_set_oil = (
        p_set_from_scaling(
            "residential oil", profile_residential, energy_totals, temporal_resolution
        )
        + heat_oil_demand
    )

    p_set_biomass = (
        p_set_from_scaling(
            "residential biomass",
            profile_residential,
            energy_totals,
            temporal_resolution,
        )
        + heat_biomass_demand
    )

    p_set_gas = (
        p_set_from_scaling(
            "residential gas", profile_residential, energy_totals, temporal_resolution
        )
        + heat_gas_demand
    )

    resdemand=p_set_oil+p_set_biomass+p_set_gas
    resdemand=resdemand.mul(temporal_resolution, axis=0)
    print("residentiallllll mapping")
    print(_mapping)
    if _mapping:
        print('residential')
        scale_res = rescale_to_mapping( resdemand*1e6/8760, _mapping)
        p_set_oil = p_set_oil * scale_res
        p_set_biomass = p_set_biomass * scale_res
        p_set_gas = p_set_gas * scale_res
        print(scale_res)

    n.madd(
        "Load",
        spatial.nodes,
        suffix=" residential oil",
        bus=spatial.oil.nodes,
        carrier="residential oil",
        p_set=p_set_oil,
    )

    # TODO: check 8760 compatibility with different snapshot settings
    co2 = p_set_oil.sum(axis=1).sum() * costs.at["oil", "CO2 intensity"]/8760
    print("residential oil")
    print(p_set_oil.sum(axis=1))
    print(p_set_oil.sum(axis=1).sum())

    n.add(
        "Load",
        "residential oil emissions",
        bus="co2 atmosphere",
        carrier="oil emissions",
        p_set=-co2,
    )
    n.madd(
        "Load",
        spatial.nodes,
        suffix=" residential biomass",
        bus=spatial.biomass.nodes,
        carrier="residential biomass",
        p_set=p_set_biomass,
    )

    n.madd(
        "Load",
        spatial.nodes,
        suffix=" residential gas",
        bus=spatial.gas.nodes,
        carrier="residential gas",
        p_set=p_set_gas,
    )

    # TODO: check 8760 compatibility with different snapshot settings
    co2 = p_set_gas.sum(axis=1).sum() * costs.at["gas", "CO2 intensity"]/8760

    n.add(
        "Load",
        "residential gas emissions",
        bus="co2 atmosphere",
        carrier="gas emissions",
        p_set=-co2,
    )

    for country in countries:
        rem_heat_demand = (
            energy_totals.loc[country, "total residential space"]
            + energy_totals.loc[country, "total residential water"]
            - energy_totals.loc[country, "residential heat biomass"]
            - energy_totals.loc[country, "residential heat oil"]
            - energy_totals.loc[country, "residential heat gas"]
        )

        heat_buses = (n.loads_t.p_set.filter(regex="heat").filter(like=country)).columns

        safe_division = safe_divide(
            n.loads_t.p_set.filter(like=country)[heat_buses],
            n.loads_t.p_set.filter(like=country)[heat_buses].sum().sum(),
        )
        n.loads_t.p_set.loc[:, heat_buses] = np.where(
            ~np.isnan(safe_division),
            (safe_division * rem_heat_demand * 1e6).div(temporal_resolution, axis=0),
            0.0,
        )

    # Revise residential electricity demand
    buses = n.buses[n.buses.carrier == "AC"].index.intersection(n.loads_t.p_set.columns)

    profile_pu = normalize_by_country(n.loads_t.p_set[buses]).fillna(0)
    n.loads_t.p_set.loc[:, buses] = p_set_from_scaling(
        "electricity residential", profile_pu, energy_totals, temporal_resolution
    )


def add_electricity_distribution_grid(n, costs):
    logger.info("Adding electricity distribution network")
    nodes = pop_layout.index

    n.madd(
        "Bus",
        nodes + " low voltage",
        location=nodes,
        carrier="low voltage",
        unit="MWh_el",
    )

    n.madd(
        "Link",
        nodes + " electricity distribution grid",
        bus0=nodes,
        bus1=nodes + " low voltage",
        p_nom_extendable=True,
        p_min_pu=-1,
        carrier="electricity distribution grid",
        efficiency=1,
        lifetime=costs.at["electricity distribution grid", "lifetime"],
        capital_cost=costs.at["electricity distribution grid", "fixed"],
    )

    # deduct distribution losses from electricity demand as these are included in total load
    # https://nbviewer.org/github/Open-Power-System-Data/datapackage_timeseries/blob/2020-10-06/main.ipynb
    if (
        efficiency := options["transmission_efficiency"]
        .get("electricity distribution grid", {})
        .get("efficiency_static")
    ):
        logger.info(
            f"Deducting distribution losses from electricity demand: {np.around(100*(1-efficiency), decimals=2)}%"
        )
        n.loads_t.p_set.loc[:, n.loads.carrier == "AC"] *= efficiency

    # move AC loads to low voltage buses
    ac_loads = n.loads.index[n.loads.carrier == "AC"]
    n.loads.loc[ac_loads, "bus"] += " low voltage"

    # move industry, rail transport, agriculture and services electricity to low voltage
    loads = n.loads.index[n.loads.carrier.str.contains("electricity")]
    n.loads.loc[loads, "bus"] += " low voltage"

    bevs = n.links.index[n.links.carrier == "BEV charger"]
    n.links.loc[bevs, "bus0"] += " low voltage"

    v2gs = n.links.index[n.links.carrier == "V2G"]
    n.links.loc[v2gs, "bus1"] += " low voltage"

    hps = n.links.index[n.links.carrier.str.contains("heat pump")]
    n.links.loc[hps, "bus0"] += " low voltage"

    rh = n.links.index[n.links.carrier.str.contains("resistive heater")]
    n.links.loc[rh, "bus0"] += " low voltage"

    mchp = n.links.index[n.links.carrier.str.contains("micro gas")]
    n.links.loc[mchp, "bus1"] += " low voltage"

    if options.get("solar_rooftop", False):
        if isinstance(options["solar_rooftop"], dict):
            enable_solar_rooftop = options["solar_rooftop"]["enable"]
            solar_opts = options["solar_rooftop"]
        else:
            enable_solar_rooftop = True
            solar_opts = {}

    if enable_solar_rooftop:
        # set existing solar to cost of utility cost rather the 50-50 rooftop-utility
        solar = n.generators.index[n.generators.carrier == "solar"]
        n.generators.loc[solar, "capital_cost"] = costs.at["solar-utility", "fixed"]

        if solar_opts.get("use_building_size"):
            solar_logger = "building_size"

            solar_rooftop_layout = pd.concat(
                [
                    pd.read_csv(snakemake.input[fn], index_col=0)
                    for fn in snakemake.input.keys()
                    if "solar_rooftop_layout" in fn
                ]
            )
            solar_rooftop_layout = solar_rooftop_layout["usefull_area"].rename(
                index=lambda x: x + " solar"
            )

            potential = (
                solar_opts.get("kW_per_m2", 0.1)
                * 1e-3  # kW to MW
                * solar_rooftop_layout
            )

        else:
            solar_logger = "population distribution"
            pop_solar = pop_layout.total.rename(index=lambda x: x + " solar")

            # add max solar rooftop potential assuming 0.1 kW/m2 and 20 m2/person,
            # i.e. 2 kW/person (population data is in thousands of people) so we get MW
            potential = (
                solar_opts.get("kW_per_m2", 0.1)
                * solar_opts.get("m2_per_person", 20)
                * pop_solar
            )

        logger.info(
            f"Adding solar rooftop technology with potential based on {solar_logger}"
        )
        print("example solar names:", list(solar[:]))
        print("potential has DC solar?", any("-DC" in s for s in potential.index))
        print("missing examples:", list(solar.difference(potential.index)[:]))
        # Handle missing potentials caused by geometry mismatches 
        missing = solar.difference(potential.index)
        if len(missing):
            logger.warning(
                f"missing solar potentials for {len(missing)} entries; setting p_nom_max=0. "
                f"missing={list(missing)}"
            )

        p_nom_max_ = potential.reindex(solar).fillna(0.0)

        n.madd(
            "Generator",
            solar,
            suffix=" rooftop",
            bus=n.generators.loc[solar, "bus"] + " low voltage",
            carrier="solar rooftop",
            p_nom_extendable=True,
            p_nom_max=p_nom_max_,#potential.loc[solar],
            marginal_cost=n.generators.loc[solar, "marginal_cost"],
            capital_cost=costs.at["solar-rooftop", "fixed"],
            efficiency=n.generators.loc[solar, "efficiency"],
            p_max_pu=n.generators_t.p_max_pu[solar],
            lifetime=costs.at["solar-rooftop", "lifetime"],
        )

    if options.get("home_battery", False):
        logger.info("Adding home battery technology")
        n.add("Carrier", "home battery")

        n.madd(
            "Bus",
            nodes + " home battery",
            location=nodes,
            carrier="home battery",
            unit="MWh_el",
        )

        n.madd(
            "Store",
            nodes + " home battery",
            bus=nodes + " home battery",
            location=nodes,
            e_cyclic=True,
            e_nom_extendable=True,
            carrier="home battery",
            capital_cost=costs.at["home battery storage", "fixed"],
            lifetime=costs.at["battery storage", "lifetime"],
        )

        n.madd(
            "Link",
            nodes + " home battery charger",
            bus0=nodes + " low voltage",
            bus1=nodes + " home battery",
            carrier="home battery charger",
            efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
            capital_cost=costs.at["home battery inverter", "fixed"],
            p_nom_extendable=True,
            lifetime=costs.at["battery inverter", "lifetime"],
        )

        n.madd(
            "Link",
            nodes + " home battery discharger",
            bus0=nodes + " home battery",
            bus1=nodes + " low voltage",
            carrier="home battery discharger",
            efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
            marginal_cost=options["marginal_cost_storage"],
            p_nom_extendable=True,
            lifetime=costs.at["battery inverter", "lifetime"],
        )


# def add_co2limit(n, Nyears=1.0, limit=0.0):
#     print("Adding CO2 budget limit as per unit of 1990 levels of", limit)

#     countries = n.buses.country.dropna().unique()

#     sectors = emission_sectors_from_opts(opts)

#     # convert Mt to tCO2
#     co2_totals = 1e6 * pd.read_csv(snakemake.input.co2_totals_name, index_col=0)

#     co2_limit = co2_totals.loc[countries, sectors].sum().sum()

#     co2_limit *= limit * Nyears

#     n.add(
#         "GlobalConstraint",
#         "CO2Limit",
#         carrier_attribute="co2_emissions",
#         sense="<=",
#         constant=co2_limit,
#     )


def add_custom_water_cost(n):
    for country in countries:
        water_costs = pd.read_csv(
            os.path.join(
                BASE_DIR,
                "resources/custom_data/{}_water_costs.csv".format(country),
                sep=",",
                index_col=0,
            )
        )
        water_costs = water_costs.filter(like=country, axis=0).loc[spatial.nodes]
        electrolysis_links = n.links.filter(like=country, axis=0).filter(
            like="lectrolysis", axis=0
        )

        elec_index = n.links[
            (n.links.carrier == "H2 Electrolysis")
            & (n.links.bus0.str.contains(country))
        ].index
        n.links.loc[elec_index, "marginal_cost"] = water_costs.values
        # n.links.filter(like=country, axis=0).filter(like='lectrolysis', axis=0)["marginal_cost"] = water_costs.values
        # n.links.filter(like=country, axis=0).filter(like='lectrolysis', axis=0).apply(lambda x: water_costs[x.index], axis=0)
        # print(n.links.filter(like=country, axis=0).filter(like='lectrolysis', axis=0).marginal_cost)


def add_rail_transport(n, costs):
    p_set_elec = nodal_energy_totals.loc[spatial.nodes, "electricity rail"]
    p_set_oil = (nodal_energy_totals.loc[spatial.nodes, "total rail"]) - p_set_elec
    n.madd(
        "Load",
        spatial.nodes,
        suffix=" rail transport oil",
        bus=spatial.oil.nodes,
        carrier="rail transport oil",
        p_set=p_set_oil * 1e6 / 8760,
    )

    n.madd(
        "Load",
        spatial.nodes,
        suffix=" rail transport electricity",
        bus=spatial.nodes,
        carrier="rail transport electricity",
        p_set=p_set_elec * 1e6 / 8760,
    )


def get_capacities_from_elec(n, carriers, component):
    """
    Gets capacities and efficiencies for {carrier} in n.{component} that were
    previously assigned in add_electricity.
    """
    component_list = ["generators", "storage_units", "links", "stores"]
    component_dict = {name: getattr(n, name) for name in component_list}
    e_nom_carriers = ["stores"]
    nom_col = {x: "e_nom" if x in e_nom_carriers else "p_nom" for x in component_list}
    eff_col = "efficiency"

    capacity_dict = {}
    efficiency_dict = {}
    node_dict = {}
    for carrier in carriers:
        capacity_dict[carrier] = component_dict[component].query("carrier in @carrier")[
            nom_col[component]
        ]
        efficiency_dict[carrier] = component_dict[component].query(
            "carrier in @carrier"
        )[eff_col]
        node_dict[carrier] = component_dict[component].query("carrier in @carrier")[
            "bus"
        ]

    return capacity_dict, efficiency_dict, node_dict


def remove_elec_base_techs(n):
    """
    Remove conventional generators (e.g. OCGT, oil) build in electricity-only network,
    since they're re-added here using links.
    """
    conventional_generators = options.get("conventional_generation", {})
    to_remove = pd.Index(conventional_generators.keys())
    # remove only conventional_generation carriers present in the network
    to_remove = pd.Index(
        snakemake.params.electricity.get("conventional_carriers", [])
    ).intersection(to_remove)

    if to_remove.empty:
        return

    logger.info(f"Removing Generators with carrier {list(to_remove)}")
    names = n.generators.index[n.generators.carrier.isin(to_remove)]
    for name in names:
        n.remove("Generator", name)
    n.carriers.drop(to_remove, inplace=True, errors="ignore")


def remove_carrier_related_components(n, carriers_to_drop):
    """
    Removes carrier related components, such as "Carrier", "Generator", "Link", "Store", and "Storage Unit"
    """
    # remove carriers
    n.carriers.drop(carriers_to_drop, inplace=True, errors="ignore")

    # remove buses, generators, stores, and storage units with carrier to remote
    for c in n.iterate_components(["Bus", "Generator", "Store", "StorageUnit"]):
        logger.info(f"Removing {c.list_name} with carrier {list(carriers_to_drop)}")
        names = c.df.index[c.df.carrier.isin(carriers_to_drop)]
        if c.name == "Bus":
            buses_to_remove = names
        n.mremove(c.name, names)

    # remove links connected to buses that were removed
    links_to_remove = n.links.query(
        "bus0 in @buses_to_remove or bus1 in @buses_to_remove or bus2 in @buses_to_remove or bus3 in @buses_to_remove or bus4 in @buses_to_remove"
    ).index
    logger.info(
        f"Removing links with carrier {list(n.links.loc[links_to_remove].carrier.unique())}"
    )
    n.mremove("Link", links_to_remove)

def apply_domestic_share_correction(domestic: pd.Series,
                                    international: pd.Series,
                                    *,
                                    enabled: bool,
                                    domestic_share: float,
                                    label: str = "") -> pd.Series:
    """
    If enabled and domestic_share>0:
      - build total = domestic + international
      - where domestic is <=0 but total>0, set domestic = domestic_share * total
    Returns corrected domestic series (same index as inputs).
    """
    dom = domestic.fillna(0.0).astype(float)
    intl = international.fillna(0.0).astype(float)

    if not enabled:
        return dom

    total = dom + intl
    m = (dom <= 0.0) & (total > 0.0)
    dom = dom.copy()
    if domestic_share != 0:
        dom.loc[m] = domestic_share * total.loc[m]
        if enabled:
            logger.info(
                f"[{label}] domestic share correction applied (share={domestic_share}). "
                f"replaced={int(m.sum())} entries."
            )
    return dom



def enforce_caps_and_re_potentials(
    n,
    *,
    re_caps_gw=None,
    group_map=None,
    carriers_check=None,
    set_min_equal_nom=True,
    raise_on_any_inf=False,
):
    """
    1) For non-extendable generators: set p_nom_max = p_nom (and optionally p_nom_min = p_nom).
    2) For extendable renewable generators: cap total p_nom_max per carrier-group to user caps (GW),
       by scaling node-level p_nom_max proportionally.
    3) Raise if any targeted extendable RE has inf/NaN p_nom_max afterwards.

    Parameters
    ----------
    n : pypsa.Network
    re_caps_gw : dict or None
        e.g. {"solar": 1000, "onwind": 800, "offwind": 300, "solar rooftop": 500}
        Values are in GW.
    group_map : dict or None
        e.g. {"offwind": ["offwind-ac", "offwind-dc"]}
        If key not in group_map, it is treated as a carrier name itself.
    carriers_check : list[str] or None
        Which carriers (or group keys) should be checked for finite p_nom_max on extendables.
        If None, checks keys from re_caps_gw.
    set_min_equal_nom : bool
        If True, sets p_nom_min = p_nom for non-extendables (if column exists).
    raise_on_any_inf : bool
        If True, raise if ANY generator has inf/NaN p_nom_max at the end (not just targeted extendables).
    """

    gens = n.generators

    # ---------- (A) Clean fixed assets ----------
    fixed = ~gens.p_nom_extendable.astype(bool)

    gens.loc[fixed, "p_nom"] = (
        pd.to_numeric(gens.loc[fixed, "p_nom"], errors="coerce")
        .fillna(0.0)
    )

    gens.loc[fixed, "p_nom_max"] = gens.loc[fixed, "p_nom"]

    if set_min_equal_nom and "p_nom_min" in gens.columns:
        gens.loc[fixed, "p_nom_min"] = gens.loc[fixed, "p_nom"]

    # ---------- (B) Global RE caps for extendables ----------
    if group_map is None:
        group_map = {"offwind": ["offwind-ac", "offwind-dc"]}

    def carriers_for(key):
        return group_map.get(key, [key])

    if re_caps_gw:
        for key, cap_gw in re_caps_gw.items():
            carriers = carriers_for(key)
            cap_mw = float(cap_gw) * 1e3

            # scale == 0 → interpret as "no cap"
            if cap_mw <= 0:
                print(f"[RE cap] {key}: disabled (cap = 0)")
                continue


            m_ext = gens.carrier.isin(carriers) & gens.p_nom_extendable.astype(bool)
            if not m_ext.any():
                continue

            pmax = pd.to_numeric(gens.loc[m_ext, "p_nom_max"], errors="coerce")
            pmax = pmax.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            total = float(pmax.sum())

            print(
                f"[RE cap] {key}: before = {total/1e3:.2f} GW | "
                f"target = {cap_mw/1e3:.2f} GW"
            )

            if total <= 0.0:
                gens.loc[m_ext, "p_nom_max"] = 0.0

            elif total <= cap_mw:
                gens.loc[m_ext, "p_nom_max"] = pmax.values

            else:
                scale = cap_mw / total
                gens.loc[m_ext, "p_nom_max"] = (pmax * scale).values

            # safety: never cap below already-built optimized capacity
            if "p_nom_opt" in gens.columns:
                gens.loc[m_ext, "p_nom_max"] = np.maximum(
                    gens.loc[m_ext, "p_nom_max"].astype(float),
                    pd.to_numeric(gens.loc[m_ext, "p_nom_opt"], errors="coerce")
                      .fillna(0.0)
                      .astype(float),
                )

            final_sum = gens.loc[m_ext, "p_nom_max"].sum()
            print(f"[RE cap] {key}: after  = {final_sum/1e3:.2f} GW\n")

    # ---------- (C) Raise if inf/NaN remains where it matters ----------
    if carriers_check is None:
        carriers_check = list((re_caps_gw or {}).keys())

    if carriers_check:
        carriers_flat = []
        for key in carriers_check:
            carriers_flat += carriers_for(key)

        m_chk = gens.p_nom_extendable.astype(bool) & gens.carrier.isin(carriers_flat)

        pmax_chk = pd.to_numeric(gens.loc[m_chk, "p_nom_max"], errors="coerce")
        bad = ~np.isfinite(pmax_chk.values)

        if bad.any():
            bad_rows = gens.loc[m_chk].iloc[np.where(bad)[0]][
                ["carrier", "bus", "p_nom_extendable", "p_nom_max"]
            ]
            raise ValueError(
                "Found non-finite p_nom_max (inf/NaN) for extendable renewables after cap enforcement.\n"
                f"Sample rows:\n{bad_rows.head(30)}"
            )

    if raise_on_any_inf:
        pmax_all = pd.to_numeric(gens["p_nom_max"], errors="coerce")
        bad_all = ~np.isfinite(pmax_all.values)

        if bad_all.any():
            bad_rows = gens.iloc[np.where(bad_all)[0]][
                ["carrier", "bus", "p_nom_extendable", "p_nom_max"]
            ]
            raise ValueError(
                "Found non-finite p_nom_max (inf/NaN) in generators after enforcement.\n"
                f"Sample rows:\n{bad_rows.head(30)}"
            )

    return n
def _fill_nan_store_p_nom(n, hours_default=24.0, pnom_floor=1.0, make_extendable=True):
    if n.stores.empty:
        return

    # ensure required columns exist
    if "p_nom" not in n.stores.columns:
        n.stores["p_nom"] = np.nan
    if "p_nom_extendable" not in n.stores.columns:
        n.stores["p_nom_extendable"] = False

    pnom = pd.to_numeric(n.stores["p_nom"], errors="coerce")
    nan_mask = pnom.isna()
    if nan_mask.any():
        enom = pd.to_numeric(
            n.stores.get("e_nom", pd.Series(np.nan, index=n.stores.index)),
            errors="coerce",
        )
        default_pnom = (enom / float(hours_default)).clip(lower=float(pnom_floor)).fillna(float(pnom_floor))
        n.stores.loc[nan_mask, "p_nom"] = default_pnom.loc[nan_mask].values

        if make_extendable:
            n.stores.loc[nan_mask, "p_nom_extendable"] = True

    # CRITICAL: enforce bool dtype (prevents NetCDF mixed-type object columns)
    n.stores["p_nom_extendable"] = n.stores["p_nom_extendable"].fillna(False).astype(bool)


if __name__ == "__main__":
    if "snakemake" not in globals():
        # from helper import mock_snakemake #TODO remove func from here to helper script
        snakemake = mock_snakemake(
            "prepare_sector_network",
            simpl="",
            clusters="4",
            ll="c1",
            opts="Co2L-4H",
            planning_horizons="2030",
            sopts="144H",
            discountrate=0.071,
            demand="AB",
        )

    # Load population layout
    pop_layout = pd.read_csv(snakemake.input.clustered_pop_layout, index_col=0)

    # Load all sector wildcards
    options = snakemake.config["sector"]
    load = snakemake.config["load_options"]['scale']

    # Load input network
    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

    # Fetch the country list from the network
    # countries = list(n.buses.country.unique())
    countries = snakemake.config["countries"]
    # Locate all the AC buses
    nodes = n.buses[
        n.buses.carrier == "AC"
    ].index  # TODO if you take nodes from the index of buses of n it's more than pop_layout
    # clustering of regions must be double checked.. refer to regions onshore

    # Add location. TODO: move it into pypsa-earth
    n.buses.location = n.buses.index

    # Set carrier of AC loads
    n.loads.loc[nodes, "carrier"] = "AC"

    Nyears = n.snapshot_weightings.generators.sum() / 8760

    # Fetch wildcards
    investment_year = int(snakemake.wildcards.planning_horizons[-4:])
    demand_sc = snakemake.wildcards.demand  # loading the demand scenrario wildcard

    # Prepare the costs dataframe
    costs = prepare_costs(
        snakemake.input.costs,
        snakemake.params.costs["USD2013_to_EUR2013"],
        snakemake.params.costs["fill_values"],
        Nyears,
    )

    # Define spatial for biomass and co2. They require the same spatial definition
    spatial = define_spatial(pop_layout.index, options)

    if snakemake.config["foresight"] in ["myopic", "perfect"]:
        add_lifetime_wind_solar(n, costs)

    # TODO logging

    nodal_energy_totals = pd.read_csv(
        snakemake.input.nodal_energy_totals,
        index_col=0,
        keep_default_na=False,
        na_values=[""],
    )
    energy_totals = pd.read_csv(
        snakemake.input.energy_totals,
        index_col=0,
        keep_default_na=False,
        na_values=[""],
    )
    # Get the data required for land transport
    # TODO Leon, This contains transport demand, right? if so let's change it to transport_demand?
    transport = pd.read_csv(snakemake.input.transport, index_col=0, parse_dates=True)

    avail_profile = pd.read_csv(
        snakemake.input.avail_profile, index_col=0, parse_dates=True
    )
    dsm_profile = pd.read_csv(
        snakemake.input.dsm_profile, index_col=0, parse_dates=True
    )
    nodal_transport_data = pd.read_csv(  # TODO This only includes no. of cars, change name to something descriptive?
        snakemake.input.nodal_transport_data, index_col=0
    )

    # Load data required for the heat sector
    heat_demand = pd.read_csv(
        snakemake.input.heat_demand, index_col=0, header=[0, 1], parse_dates=True
    ).fillna(0)
    # Ground-sourced heatpump coefficient of performance
    gshp_cop = pd.read_csv(
        snakemake.input.gshp_cop, index_col=0, parse_dates=True
    )  # only needed with heat dep. hp cop allowed from config
    # TODO add option heat_dep_hp_cop to the config

    # Air-sourced heatpump coefficient of performance
    ashp_cop = pd.read_csv(
        snakemake.input.ashp_cop, index_col=0, parse_dates=True
    )  # only needed with heat dep. hp cop allowed from config

    # Solar thermal availability profiles
    solar_thermal = pd.read_csv(
        snakemake.input.solar_thermal, index_col=0, parse_dates=True
    )
    gshp_cop = pd.read_csv(snakemake.input.gshp_cop, index_col=0, parse_dates=True)

    # Share of district heating at each node
    district_heat_share = pd.read_csv(snakemake.input.district_heat_share, index_col=0)

    # Load data required for aviation and navigation
    # TODO follow the same structure as land transport and heat

    # Load industry demand data
    # --- Scaling for transport (drop-in fix) ---

    hours = n.snapshot_weightings.generators
    H_yr = float(hours.sum())

    # ----- PORTS (shipping H2 share) -----
    ports = pd.read_csv(snakemake.input.ports, keep_default_na=False)
    ports = ports[ports.country.isin(countries)]

    gadm_layer_id = snakemake.config["build_shape_options"]["gadm_layer_id"]
    ports = locate_bus(
        ports, countries, gadm_layer_id, snakemake.input.shapes_path,
        snakemake.config["cluster_options"]["alternative_clustering"],
    ).set_index(f"gadm_{gadm_layer_id}")

    ports["fraction"] = ports["fraction"] / ports["fraction"].sum() if ports["fraction"].sum() else 0.0
    shipping_hydrogen_share = get(options["shipping_hydrogen_share"], demand_sc + "_" + str(investment_year))
    if snakemake.config["sector"]["international_bunkers"]:
        all_navigation = ["total domestic navigation", "total international navigation"]
    else:
        all_navigation = ["total domestic navigation"]

    navigation_TWh = float(energy_totals.loc[countries, all_navigation].sum(axis=1).sum())
    efficiency = options["shipping_average_efficiency"] / costs.at["fuel cell", "efficiency"]

    # constant MW per port
    ports["p_set"] = ports["fraction"] * shipping_hydrogen_share * navigation_TWh * efficiency * 1e6 / 8760.0
    print('port')
    print(shipping_hydrogen_share)
    navigation_demand = (
        energy_totals.loc[countries, all_navigation].sum(axis=1).sum()  # * 1e6 / 8760
    )

    if shipping_hydrogen_share < 1:
        shipping_oil_share = 1 - shipping_hydrogen_share
        print(f"shipping_oil_share = {shipping_oil_share}")
        print(f"navigation_demand = {navigation_demand}")
        print(ports["fraction"])
        ports["p_set"] = ports["fraction"].apply(
            lambda frac: shipping_oil_share * frac * navigation_demand * 1e6 / 8760
        )
        print('skibidi')
        print(ports["p_set"])

    ports_total_MWh = float(ports["p_set"].sum()) * H_yr  # MW * hours -> MWh/a

    # ----- AIRPORTS (aviation) -----
    airports = pd.read_csv(snakemake.input.airports, keep_default_na=False)
    airports = airports[airports.country.isin(countries)]
    airports = locate_bus(
        airports, countries, gadm_layer_id, snakemake.input.shapes_path,
        snakemake.config["cluster_options"]["alternative_clustering"],
    ).set_index(f"gadm_{gadm_layer_id}")

    airports["fraction"] = airports["fraction"] / airports["fraction"].sum() if airports["fraction"].sum() else 0.0
    if snakemake.config["sector"]["international_bunkers"]:
        all_aviation = ["total domestic aviation", "total international aviation"]
    else:
        all_aviation = ["total domestic aviation"]
    aviation_TWh = float(energy_totals.loc[countries, all_aviation].sum(axis=1).sum())

    airports["p_set"] = airports["fraction"] * aviation_TWh * 1e6 / 8760.0  # MW
    airports_total_MWh = float(airports["p_set"].sum()) * H_yr

    # ----- LAND TRANSPORT (from network loads) -----

    # hours per snapshot (for 3-hourly snapshots -> 3, for 1-hourly -> 1)
    # best is to use snapshot weightings if you have them
    hours = n.snapshot_weightings["generators"]  # or objective, depending on your setup

    transport_total_MWh = transport.mul(hours, axis=0).to_numpy().sum()


    # ----- RAIL (prefer network loads; fallback to energy_totals) -----
    rail_carriers = ["rail transport electricity", "rail transport oil"]
    rail_cols = n.loads.index[n.loads.carrier.isin(rail_carriers)]
    rail_cols = n.loads_t.p_set.columns.intersection(rail_cols)

    if len(rail_cols) > 0:
        rail_total_MWh = float(n.loads_t.p_set[rail_cols].mul(hours, axis=0).to_numpy().sum())
    else:
        rail_total_MWh = float(nodal_energy_totals.loc[spatial.nodes, "total rail"].sum() * 1e6)  # TWh -> MWh

    # ----- Build comparison frame (all in MWh/a) -----
    p_set_df = pd.DataFrame({
        "transport": [transport_total_MWh],
    })

    print("Transport!!!!", flush=True)
    print(p_set_df, flush=True)

    # ----- Rescale land transport to mapping (mapping is MWh/a) -----
    # _mapping must be in MWh/a and p_set_df must already be in MWh/a
    _mapping = load.get("transport", {}).get(investment_year, {})

    if _mapping:
        factor_transport = rescale_to_mapping(p_set_df, _mapping)  # returns a scalar
        transport = transport * factor_transport                   # scale the time series (MW)
        print("factor=", factor_transport, flush=True)
    else:
        factor_transport = 1.0

    ##########################################################################
    ############## Functions adding different carrires and sectors ###########
    ##########################################################################

    # read existing installed capacities of generators
    if options.get("keep_existing_capacities", False):
        existing_capacities, existing_efficiencies, existing_nodes = (
            get_capacities_from_elec(
                n,
                carriers=options.get("conventional_generation").keys(),
                component="generators",
            )
        )
        print(existing_capacities)
    else:
        existing_capacities, existing_efficiencies, existing_nodes = 0, None, None

    add_co2(n, costs,Nyears)  # TODO add costs
    cap_cfg = snakemake.config["custom_RE_cap"]
    if cap_cfg.get("enable", False):

        units = cap_cfg.get("units", "GW").upper()
        caps = cap_cfg.get("caps", {}) or {}

        # convert units -> GW (function expects GW)
        if units == "MW":
            caps_gw = {k: float(v) / 1e3 for k, v in caps.items()}
        elif units == "GW":
            caps_gw = {k: float(v) for k, v in caps.items()}
        else:
            raise ValueError(f"custom_RE_cap.units must be 'MW' or 'GW', got: {units}")

        enforce_caps_and_re_potentials(
            n,
            re_caps_gw=caps_gw,
            group_map={"offwind": ["offwind-ac", "offwind-dc"]},
            carriers_check=["solar", "solar rooftop", "onwind", "offwind"],
            raise_on_any_inf=False,  # strict check is already on extendable RE carriers
        )

    # remove conventional generators built in elec-only model
    remove_elec_base_techs(n)

    add_generation(n, costs, existing_capacities, existing_efficiencies, existing_nodes)

    # remove H2 and battery technologies added in elec-only model
    remove_carrier_related_components(n, carriers_to_drop=["H2", "battery"])

    add_hydrogen(n, costs)  # TODO add costs
    
    missing_x = n.buses.loc[n.buses.index.intersection(spatial.nodes), "x"].isna()
    missing_y = n.buses.loc[n.buses.index.intersection(spatial.nodes), "y"].isna()
    missing_coords = missing_x | missing_y
    if missing_coords.any():
        print("Error: Missing coordinates detected in the following spatial nodes:")
        print(n.buses.loc[missing_coords.index[missing_coords], ["x", "y"]])
        raise ValueError("Aborting: Missing x/y coordinates for spatial nodes.")
    missing_nodes = set(spatial.nodes) - set(n.buses.index)
    if missing_nodes:
        print("Warning: The following spatial nodes are not present in n.buses:")
        print(missing_nodes)
        raise ValueError("Aborting: Some spatial.nodes are missing from n.buses.")

    add_storage(n, costs)

    H2_liquid_fossil_conversions(n, costs)

    h2_hc_conversions(n, costs)
    add_heat(n, costs)
    add_biomass(n, costs)

    # ---- tie legacy biomass to the capped solid-biomass store ----
    ix_eop = n.links.carrier.str.contains("biomass EOP", case=False)
    if ix_eop.any():
        solid_bus_idx = pd.Index(spatial.biomass.nodes)

        if len(solid_bus_idx) == 1:
            # one global store/bus -> assign the same bus to all EOP links
            n.links.loc[ix_eop, "bus0"] = solid_bus_idx[0]
        else:
            # per-node store/bus -> map node-by-node
            # links are named "<node> biomass EOP"; build a mapping from link name -> "<node> solid biomass"
            node_to_bus = pd.Series(
                data=solid_bus_idx.values,              # "<node> solid biomass"
                index=pd.Index(spatial.nodes)           # "<node>"
            )
            link_to_bus = node_to_bus.rename(
                index=lambda node: f"{node} biomass EOP"  # index becomes the link name
            )
            # assign only for links that actually exist
            common = link_to_bus.index.intersection(n.links.index)
            n.links.loc[common, "bus0"] = link_to_bus.loc[common].values

    # remove the unconstrained legacy biomass fuel store/carrier (if any)
    bad_stores = n.stores.index[n.stores.carrier == "biomass"]
    if len(bad_stores):
        n.mremove("Store", bad_stores)
    if "biomass" in n.carriers.index:
        n.carriers.drop("biomass", errors="ignore", inplace=True)
    n.buses.loc[n.buses.carrier == "biomass", "carrier"] = "solid biomass"

    # make sure solid biomass store cannot extend
    n.stores.loc[n.stores.carrier == "solid biomass", "e_nom_extendable"] = False
    # --------------------------------------------------------------


    add_industry(n, costs)

    add_shipping(n, costs)

    # Add_aviation runs with dummy data
    add_aviation(n, costs)

    # prepare_transport_data(n)

    add_land_transport(n, costs)

    # if snakemake.config["custom_data"]["transport_demand"]:
    add_rail_transport(n, costs)

    # if snakemake.config["custom_data"]["custom_sectors"]:
    add_agriculture(n, costs)
    add_residential(n, costs)
    add_services(n, costs)

    if options.get("electricity_distribution_grid", False):
        add_electricity_distribution_grid(n, costs)

    sopts = snakemake.wildcards.sopts.split("-")

    for o in sopts:
        m = re.match(r"^\d+h$", o, re.IGNORECASE)
        if m is not None:
            n = average_every_nhours(n, m.group(0))
            break

    # TODO add co2 limit here, if necessary
    # co2_limit_pu = eval(sopts[0][5:])
    # co2_limit = co2_limit_pu *
    # # Add co2 limit
    # co2_limit = 1e9
    # n.add(
    #     "GlobalConstraint",
    #     "CO2Limit",
    #     carrier_attribute="co2_emissions",
    #     sense="<=",
    #     constant=co2_limit,
    # )

    if options["dac"]:
        add_dac(n, costs)

    if snakemake.config["custom_data"]["water_costs"]:
        add_custom_water_cost(n)
    
    # apply to both generators and links
    for comp in ["generators", "links"]:
        df = getattr(n, comp)
        if "p_nom_extendable" in df.columns:
            mask = df["p_nom_extendable"].astype(bool)
            # if extendable, enforce existing capacity as minimum
            df.loc[mask, "p_nom_min"] = np.maximum(df.loc[mask, "p_nom_min"].fillna(0), df.loc[mask, "p_nom"])

    
    sanitize_carriers(n, snakemake.config)
    sanitize_locations(n)

    n.buses.loc["co2 atmosphere", ["x","y"]] = (0.0, 0.0)
    n.buses.loc["co2 stored", ["x","y"]] = (1.0, 0.0)
    n.buses.loc["process emissions", ["x","y"]] = (0.0, 1.0)

    def show_offwind_caps(n, tag):
        g = n.generators[n.generators.carrier.astype(str).str.contains("offwind", na=False)]
        print(f"\n[{tag}] offwind count={len(g)}")
        print("p_nom_max finite share:", np.isfinite(g.p_nom_max).mean())
        print(g[["bus","p_nom_max"]].sort_values("p_nom_max", ascending=False).head(10))

    show_offwind_caps(n, "Prepare sector network")
    off = n.generators[n.generators.index.str.contains("offwind", case=False)]
    off[["bus","carrier","p_nom_opt","p_nom_max","capital_cost","marginal_cost"]].head()
    print(off["p_nom_max"].value_counts().head(10))

    print(off.groupby("carrier")["p_nom_max"].agg(["count","min","max"]).sort_values("max", ascending=False))



    _fill_nan_store_p_nom(n, hours_default=24.0, pnom_floor=1.0, make_extendable=True)
    _assert_no_nans_in_timeseries(n)
    _assert_component_bounds_sane(n)
    print("[debug] sanity checks passed: no NaNs/infs, bounds look sane")

    bad = (
        n.buses.carrier.isna()
        | n.buses.carrier.astype(str).str.strip().isin(["", "-"])
    )

    if bad.any():
        print("❌ buses with invalid carrier detected:\n")
        print(n.buses.loc[bad, ["carrier"]].head(50))
        print("\nexample bad bus names:", n.buses.index[bad].tolist()[:50])

        raise RuntimeError(
            f"{bad.sum()} bus(es) have missing/blank carrier. "
            "This will corrupt energy-balance grouping — aborting."
        )


    n.export_to_netcdf(snakemake.output[0])

    # TODO changes in case of myopic oversight
