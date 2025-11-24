import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import math

# parameters
timelimit = 1200 # Increased TimeLimit to 1200 seconds (20 minutes)
instance_sizes = [5] 

# --- DATA LOADING AND PREPARATION ---

# Load data
nodes_df = pd.read_excel("customers.xlsx",sheet_name="Nodes")
request_df = pd.read_excel("customers.xlsx",sheet_name="Requests")
fleet_df = pd.read_excel("customers.xlsx",sheet_name="Fleet") 

depot = nodes_df[nodes_df['type'] == 0]
depot_id = int(depot['id'].iloc[0])
customer_id = nodes_df[nodes_df['type'] == 1]['id'].tolist()
all_nodes = nodes_df['id'].tolist()
coordinates = nodes_df.set_index("id")[["cx","cy"]].to_dict("index")

# Extract max travel time for the vehicle
MAX_TOUR_DURATION = fleet_df['max_travel_time'].iloc[0]

# Extract time window and service time data
request_map = request_df.set_index('node')[['start', 'end', 'service_time']].to_dict('index')

DEPOT_TIME_WINDOW = {'start': 0, 'end': 10000, 'service_time': 0}

earliest_time = {
    node_id: request_map.get(node_id, DEPOT_TIME_WINDOW)['start']
    for node_id in all_nodes
}
latest_time = {
    node_id: request_map.get(node_id, DEPOT_TIME_WINDOW)['end']
    for node_id in all_nodes
}
service_time = {
    node_id: request_map.get(node_id, DEPOT_TIME_WINDOW)['service_time']
    for node_id in all_nodes
}

# Euclidean distance 
def dist(i, j):
    x = coordinates[i]["cx"] - coordinates[j]["cx"]
    y = coordinates[i]["cy"] - coordinates[j]["cy"]
    distance = int(round(math.sqrt(x**2 + y**2)))
    return distance

for n in instance_sizes:

    node_ids = [depot_id] + customer_id[:n]
    n_nr = len(node_ids)

    MTZ_model = gp.Model("MTZ_VRPTW")
    MTZ_model.setParam('TimeLimit', timelimit)
    MTZ_model.setParam('OutputFlag', 1)
    # Hint Gurobi to focus on finding a feasible solution quickly
    MTZ_model.setParam('MIPFocus', 1) 

    x = MTZ_model.addVars(n_nr, n_nr, vtype=GRB.BINARY, name="x")
    for i in range(n_nr):
        x[i, i].ub = 0
    
    u = MTZ_model.addVars(n_nr, vtype=GRB.CONTINUOUS, lb=0, ub=n_nr-1, name="u")

    # --- TIME VARIABLES ---
    t_lb = [earliest_time[node_ids[i]] for i in range(n_nr)]
    t_ub = [latest_time[node_ids[i]] for i in range(n_nr)]
    t = MTZ_model.addVars(n_nr, vtype=GRB.CONTINUOUS, lb=t_lb, ub=t_ub, name="t")
    w = MTZ_model.addVars(n_nr, vtype=GRB.CONTINUOUS, lb=0, name="w")

    # objective
    MTZ_model.setObjective(gp.quicksum(dist(node_ids[i], node_ids[j]) * x[i,j]
                                     for i in range(n_nr) for j in range(n_nr)),
                         GRB.MINIMIZE)

    # constraints
    # Degree constraints (entry and exit)
    for i in range(n_nr):
        MTZ_model.addConstr(gp.quicksum(x[i,j] for j in range(n_nr) if j != i) == 1, name=f"Out_{i}")
        MTZ_model.addConstr(gp.quicksum(x[j,i] for j in range(n_nr) if j != i) == 1, name=f"In_{i}")

    # MTZ Subtour Elimination Constraints
    MTZ_model.addConstr(u[0] == 0, name="U_Depot_0")
    for i in range(1, n_nr):
        MTZ_model.addConstr(u[i] >= 1, name=f"U_Lower_{i}")
        MTZ_model.addConstr(u[i] <= n_nr - 1, name=f"U_Upper_{i}")

    for i in range(1, n_nr):
        for j in range(1, n_nr):
            if i != j:
                MTZ_model.addConstr(u[i] - u[j] + n_nr * x[i,j] <= n_nr - 1, name=f"MTZ_{i}_{j}")

    # --- TIME WINDOW CONSTRAINTS ---

    # Define a Big M
    M = sum(dist(node_ids[i], node_ids[j]) for i in range(n_nr) for j in range(n_nr)) + max(latest_time.values())
    M = max(M, 10000000)

    # 1. Enforce wait time at the depot (node 0) is 0
    MTZ_model.addConstr(w[0] == 0, name="Depot_Wait_Time")
    
    for i in range(n_nr):
        node_id_i = node_ids[i]
        
        # 2. Service Start Time Constraint
        MTZ_model.addConstr(t[i] + w[i] <= latest_time[node_id_i], name=f"Latest_Departure_{i}")
        
        S_i = service_time[node_id_i]
        
        for j in range(n_nr):
            if i != j:
                node_id_j = node_ids[j]
                Travel_Time_ij = dist(node_id_i, node_id_j)
                
                # 3. Time Linking Constraint
                MTZ_model.addConstr(t[i] + w[i] + S_i + Travel_Time_ij - M * (1 - x[i,j]) <= t[j],
                                    name=f"Time_Link_{i}_{j}") 
                

    # --- NEW: MAX TOUR DURATION CONSTRAINT ---
    # The arrival time back at the depot (t[0] upon return) must be less than MAX_TOUR_DURATION
    # We must identify the arc returning to the depot, which is x[j, 0] = 1 for some j.
    # The return time t[0] is the maximum of (t[j] + w[j] + S_j + dist(j, 0)) for all j where x[j, 0] = 1.

    # Find the variable representing the arrival time back at the depot.
    # We can enforce t[0]_return <= MAX_TOUR_DURATION for all arcs returning to 0.
    for j in range(1, n_nr):
        # Time arrival at depot (t[0]) must be >= departure from j + travel time
        S_j = service_time[node_ids[j]]
        Travel_Time_j_0 = dist(node_ids[j], node_ids[0])

        MTZ_model.addConstr(t[j] + w[j] + S_j + Travel_Time_j_0 - M * (1 - x[j, 0]) <= MAX_TOUR_DURATION,
                            name=f"Max_Duration_{j}_0")

    MTZ_model.optimize()

    # --- SOLUTION EXTRACTION ---
    if MTZ_model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        if MTZ_model.objVal > 1e10:
             # This can happen if Gurobi terminated early without finding any solution
             print("\nModel terminated without finding a valid solution within the time limit.")
             # Re-run the infeasibility check if needed
        else:
            sol = MTZ_model.getAttr("x", x)
            time_sol = MTZ_model.getAttr("x", t)
            wait_sol = MTZ_model.getAttr("x", w)

            tour = [0] 
            current = 0
            
            print("\n--- Model Solved Successfully ---")
            print("Objective (Total Distance) =", MTZ_model.objVal)

            # Build Tour and Print Details
            tour_details = []
            
            # Start at depot (index 0)
            tour_details.append({
                "Node ID": node_ids[0],
                "Time Window": f"[{earliest_time.get(node_ids[0], '-')}, {latest_time.get(node_ids[0], '-')}]",
                "Arrival Time": time_sol[0],
                "Wait Time": wait_sol[0],
                "Service Time": service_time[node_ids[0]],
                "Departure Time": time_sol[0] + wait_sol[0] + service_time[node_ids[0]],
                "Travel Time to Next": 0 # Will be filled in the loop
            })

            for a in range(n_nr - 1):
                from_node_idx = current
                
                # Find next node
                next_node_idx = -1
                for j in range(n_nr):
                    if sol[from_node_idx, j] > 0.5:
                        next_node_idx = j
                        break

                if next_node_idx == -1:
                    break

                current = next_node_idx
                tour.append(current)

                from_node_id = node_ids[from_node_idx]
                to_node_id = node_ids[next_node_idx]
                travel_time = dist(from_node_id, to_node_id)
                
                # Update travel time from previous node
                tour_details[-1]['Travel Time to Next'] = travel_time

                # Add details for the new node (which is the current node)
                tour_details.append({
                    "Node ID": to_node_id,
                    "Time Window": f"[{earliest_time.get(to_node_id, '-')}, {latest_time.get(to_node_id, '-')}]",
                    "Arrival Time": time_sol[next_node_idx],
                    "Wait Time": wait_sol[next_node_idx],
                    "Service Time": service_time[to_node_id],
                    "Departure Time": time_sol[next_node_idx] + wait_sol[next_node_idx] + service_time[to_node_id],
                    "Travel Time to Next": 0 # Will be filled in next iteration or return to depot
                })
        
            # Add final return trip to depot
            last_customer_idx = tour[-1]
            last_customer_id = node_ids[last_customer_idx]
            tour_details[-1]['Travel Time to Next'] = dist(last_customer_id, node_ids[0])
            tour.append(0)
            
            tour_ids = [node_ids[i] for i in tour]

            # Final depot arrival (t[0])
            final_arrival_at_depot = time_sol[0]
            
            print("Tour (Node IDs) =", tour_ids)
            print(f"Max Allowed Tour Duration: {MAX_TOUR_DURATION:.2f}")
            print(f"Final Arrival at Depot: {final_arrival_at_depot:.2f}")

            # Format and display time window details
            df_results = pd.DataFrame(tour_details)
            print("\n--- Time Window Details ---")
            print(df_results[['Node ID', 'Time Window', 'Arrival Time', 'Wait Time', 'Service Time', 'Departure Time', 'Travel Time to Next']].to_string(index=False, float_format="%.2f"))

    elif MTZ_model.status == GRB.INFEASIBLE:
        print("\nModel is Infeasible. Computing Irreducible Infeasible Subsystem (IIS)...")
        MTZ_model.computeIIS()
        MTZ_model.write("mtz_vrptw_iis.ilp")
        print("IIS written to mtz_vrptw_iis.ilp. Check this file for conflicting constraints.")
        
        iis_constraints = [c.ConstrName for c in MTZ_model.getConstrs() if c.IISConstr]
        iis_vars = [v.VarName for v in MTZ_model.getVars() if v.IISLB or v.IISUB]
        print(f"Conflicting constraints in IIS: {iis_constraints}")
        print(f"Conflicting variable bounds in IIS: {iis_vars}")

    else:
        print("\nModel terminated for status code:", MTZ_model.status)
        print("Model terminated without finding a solution. Try increasing the time limit or further relaxing constraints.")