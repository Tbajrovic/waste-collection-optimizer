from __future__ import annotations

import argparse
from pathlib import Path
import math
import pandas as pd

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve a CVRP baseline using OR-Tools.")
    parser.add_argument("--nodes", type=str, default="Results/town_of_cary_nodes.csv")
    parser.add_argument("--dist", type=str, default="Results/town_of_cary_distance_matrix.csv")
    parser.add_argument("--out", type=str, default="Results/routes_optimized.csv")

    parser.add_argument("--num_vehicles", type=int, default=3)
    parser.add_argument("--vehicle_capacity", type=int, default=50)
    parser.add_argument("--demand_per_stop", type=int, default=1)

    parser.add_argument("--time_limit_s", type=int, default=10)
    args = parser.parse_args()

    nodes_path = Path(args.nodes)
    dist_path = Path(args.dist)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not nodes_path.exists():
        raise FileNotFoundError(f"Nodes file not found: {nodes_path.as_posix()}")
    if not dist_path.exists():
        raise FileNotFoundError(f"Distance matrix file not found: {dist_path.as_posix()}")

    nodes = pd.read_csv(nodes_path)
    dist_df = pd.read_csv(dist_path, index_col=0)

    node_ids = nodes["node_id"].astype(str).tolist()

    # Ensure matrix matches node order
    dist_df = dist_df.loc[node_ids, node_ids]

    # OR-Tools wants integer costs
    dist_m = dist_df.to_numpy(dtype=float)
    dist_int = (dist_m.round().astype(int)).tolist()

    depot_index = 0  # we wrote depot first in prepare_routing_data.py

    # Demand: depot has 0, each stop has constant demand (simple baseline)
    demands = [0] + [args.demand_per_stop] * (len(node_ids) - 1)

    # Create routing index manager + model
    manager = pywrapcp.RoutingIndexManager(len(node_ids), args.num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_int[from_node][to_node]

    transit_cb_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    # Demand/capacity constraint
    def demand_callback(from_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx,
        0,  # slack
        [args.vehicle_capacity] * args.num_vehicles,
        True,  # start cumul to zero
        "Capacity",
    )

    # Search parameters
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.FromSeconds(args.time_limit_s)

    solution = routing.SolveWithParameters(search_params)

    if solution is None:
        raise RuntimeError("No solution found. Try increasing time limit or vehicles/capacity.")

    # Extract routes
    rows = []
    total_distance = 0

    for v in range(args.num_vehicles):
        index = routing.Start(v)
        route = []
        route_dist = 0

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node_ids[node])
            next_index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(next_index)

            route_dist += dist_int[node][next_node]
            index = next_index

        # End node (depot)
        route.append(node_ids[manager.IndexToNode(index)])
        total_distance += route_dist

        rows.append(
            {
                "vehicle_id": v,
                "num_stops_including_depot": len(route),
                "route_distance_m": route_dist,
                "route": " -> ".join(route),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)

    print("[OK] CVRP solved.")
    print(f"[INFO] Vehicles: {args.num_vehicles}, Capacity: {args.vehicle_capacity}, Demand/stop: {args.demand_per_stop}")
    print(f"[INFO] Total distance (m): {total_distance}")
    print(f"[INFO] Saved routes to: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
