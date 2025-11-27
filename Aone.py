import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
from itertools import combinations
import matplotlib.pyplot as plt
import os


def solve_tsp_mtz(df_nodes, n_customers, time_limit):

    
    nodes = df_nodes[df_nodes['id'] <= n_customers].copy()
    
    coords = nodes[['cx', 'cy']].values
    n_nodes = len(coords)
    
    dist_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                dist_matrix[i][j] = round(np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2))
                
    

    model = gp.Model("TSP_MTZ")
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0) 
    

    x = model.addVars(n_nodes, n_nodes, vtype=GRB.BINARY, name="x")
    

    u = model.addVars(range(1, n_nodes), lb=1.0, ub=n_nodes - 1.0, vtype=GRB.CONTINUOUS, name="u")

    model.addConstrs((gp.quicksum(x[i, j] for j in range(n_nodes) if j != i) == 1 
                      for i in range(n_nodes)), name="Leave")
    

    model.addConstrs((gp.quicksum(x[i, j] for i in range(n_nodes) if i != j) == 1 
                      for j in range(n_nodes)), name="Enter")

    model.addConstrs(
        (u[i] - u[j] + n_nodes * x[i, j] <= n_nodes - 1 
         for i in range(1, n_nodes) for j in range(1, n_nodes) if i != j), 
        name="MTZ"
    )

    model.setObjective(
        gp.quicksum(dist_matrix[i][j] * x[i, j] for i in range(n_nodes) for j in range(n_nodes) if i != j), 
        GRB.MINIMIZE
    )
    
    model.optimize()
    
    
    result = {
        "status": None,
        "n_nodes": n_nodes,
        "objective_value": None,
        "best_bound": None,
        "tour": [],
        "runtime": model.Runtime
    }
    
    if model.status == GRB.OPTIMAL:
        result["status"] = "Optimal"
        result["objective_value"] = model.objVal
        result["gap"] = 0.0
        
        
    elif model.status == GRB.TIME_LIMIT and model.SolCount > 0:
        result["status"] = "Time Limit"
        result["objective_value"] = model.objVal  
        result["gap"] = model.MIPGap * 100       
        

        
        
    return result

def solve_tsp_dfj(df_nodes, n_customers, time_limit):

    start_time = time.time()

    nodes = df_nodes[df_nodes['id'] <= n_customers].copy()
    coords = nodes[['cx', 'cy']].values
    n_nodes = len(coords)
    
    dist_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                d = np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)
                dist_matrix[i][j] = round(d)

    # From this line is lot of help form AI, in this function, I was having a problem how to handel errors and time limits
    try:
           
            model = gp.Model("TSP_DFJ")
            model.setParam('OutputFlag', 0)
            
            elapsed = time.time() - start_time
            if elapsed > time_limit:
                return {
                    "status": "Time Limit Exceeded (Data Prep)", 
                    "n_nodes": n_nodes, 
                    "runtime": time_limit, 
                    "objective_value": "N/A"
                }

            model.setParam('TimeLimit', max(0, time_limit - elapsed))
            
            x = model.addVars(n_nodes, n_nodes, vtype=GRB.BINARY, name="x")
            
            model.addConstrs((gp.quicksum(x[i, j] for j in range(n_nodes) if j != i) == 1 for i in range(n_nodes)), name="Leave")
            model.addConstrs((gp.quicksum(x[i, j] for i in range(n_nodes) if i != j) == 1 for j in range(n_nodes)), name="Enter")
            

            
            for r in range(2, n_nodes):
                for subset in combinations(range(n_nodes), r):
                    
                  
                    if (time.time() - start_time) > time_limit:
                        return {
                            "status": "Time Limit Exceeded (Building)", 
                            "n_nodes": n_nodes, 
                            "runtime": time_limit,
                            "objective_value": "N/A"
                        }
                    
                    
                    model.addConstr(
                        gp.quicksum(x[i, j] for i in subset for j in subset if i != j) <= len(subset) - 1
                    )

      
            elapsed_so_far = time.time() - start_time
            if elapsed_so_far > time_limit:
                return {
                    "status": "Time Limit Exceeded (Pre-Solve)", 
                    "n_nodes": n_nodes, 
                    "runtime": time_limit, 
                    "objective_value": "N/A"
                }
                
            model.setParam('TimeLimit', max(0, time_limit - elapsed_so_far))
            model.setObjective(gp.quicksum(dist_matrix[i][j] * x[i, j] for i in range(n_nodes) for j in range(n_nodes) if i != j), GRB.MINIMIZE)
            
            model.optimize()
            
            total_runtime = time.time() - start_time
            
            if model.status == GRB.OPTIMAL:
                return {"status": "Optimal", "n_nodes": n_nodes, "objective_value": model.objVal, "runtime": total_runtime}
            elif model.status == GRB.TIME_LIMIT:
                return {"status": "Time Limit Exceeded (Solver)", "n_nodes": n_nodes, "objective_value": "N/A", "runtime": total_runtime}
            else:
                return {"status": f"Other Status ({model.status})", "n_nodes": n_nodes, "objective_value": "N/A", "runtime": total_runtime}

    except gp.GurobiError as e:
            return {
                "status": f"Gurobi Error: {str(e)}", 
                "n_nodes": n_nodes, 
                "objective_value": "N/A", 
                "runtime": time.time() - start_time
            }
    except MemoryError:
            return {
                "status": "System Memory Error", 
                "n_nodes": n_nodes, 
                "objective_value": "N/A", 
                "runtime": time.time() - start_time
            }
    except Exception as e:
            return {
                "status": f"Error: {str(e)}", 
                "n_nodes": n_nodes, 
                "objective_value": "N/A", 
                "runtime": time.time() - start_time
            }
    
def test_mtz(df_nodes, instances, time_limit):
    
    
    mtz_results = []
    
    for n in instances:
        
        res_mtz = solve_tsp_mtz(df_nodes, n, time_limit)
        mtz_time = res_mtz['runtime']
        mtz_obj = res_mtz['objective_value'] if res_mtz['objective_value'] is not None else "N/A"
        mtz_status = res_mtz['status']
        mtz_bound = res_mtz.get('best_bound', None)
    
        mtz_results.append({
            "Customers": n,
            "Formulation": "MTZ",
            "Status": mtz_status,
            "Time_s": round(mtz_time, 2),
            "Objective": mtz_obj,
            "Best_Bound": round(mtz_bound, 2) if mtz_bound is not None else None
        })
    

    df_mtz_results = pd.DataFrame(mtz_results)
    
    os.makedirs(r"Results\MTZ", exist_ok=True)
    df_mtz_results.to_csv(r"Results\MTZ\mtz_results.csv", index=False)
    
    
    return df_mtz_results

def test_dfj(df_nodes, instances, time_limit):
    
    dfj_naive_results = []

    for n in instances:
        
        res_dfj = solve_tsp_dfj(df_nodes, n, time_limit)
        dfj_time = res_dfj['runtime']
        dfj_obj = res_dfj['objective_value'] if res_dfj['objective_value'] != "N/A" else "N/A"
        dfj_status = res_dfj['status']
        
        
        dfj_naive_results.append({
            "Customers": n,
            "Formulation": "DFJ",
            "Status": dfj_status,
            "Time_s": round(dfj_time, 2),
            "Objective": dfj_obj
        })
    
    df_dfj_results = pd.DataFrame(dfj_naive_results)
    
    os.makedirs(r"Results\DFJ", exist_ok=True)
    df_dfj_results.to_csv(r"Results\DFJ\dfj_results.csv", index=False)
    

    
    return df_dfj_results

def find_subtours(nodes,edges):
    
    next_node = {}
    for u, v in edges:
        next_node[u] = v
        
    visited = [False]* len(nodes)
    
    tours = []
    
    for i in range(len(nodes)):
        if not visited[i]:
            tour = []
            current = i
            while not visited[current]:
                visited[current] = True
                tour.append(current)
                current = next_node[current]
            tours.append(tour)
            
    return tours
        
def solve_tsp_dfj_improve(df_nodes, n_customers, time_limit):
    
    nodes = df_nodes[df_nodes['id'] <= n_customers].copy()
    
    coords = nodes[['cx', 'cy']].values
    n_nodes = len(coords)
    
    dist_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                 dist_matrix[i][j] = round(np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2))
                
    model = gp.Model("TSP_DFJ")
    
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)
    
    
    x = model.addVars(n_nodes, n_nodes, vtype=GRB.BINARY, name="x")
    
    model.addConstrs((gp.quicksum(x[i, j] for j in range(n_nodes) if j != i) == 1 for i in range(n_nodes)), name="Leave")
    model.addConstrs((gp.quicksum(x[i, j] for i in range(n_nodes) if i != j) == 1 for j in range(n_nodes)), name="Enter")
    
    
    model.setObjective(gp.quicksum(dist_matrix[i][j] * x[i, j] for i in range(n_nodes) for j in range(n_nodes) if i != j), GRB.MINIMIZE)
    
    iteration_results = {}
    iteration = 0

    
    while True:
        iteration += 1
        model.optimize()
        
        if model.status != GRB.OPTIMAL:
            break
    
        
        edges = []
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and x[i, j].X > 0.5:
                    edges.append((i, j))
                    
        subtours = find_subtours(nodes, edges)
        
        
        if len(subtours) == 1:
            break
        
        for subtour in subtours:
            model.addConstr(
            gp.quicksum(x[i, j] for i in subtour for j in subtour if i != j) <= len(subtour) - 1, 
                name=f"SubtourElim_Iter{iteration}"
            ) # This line is with help of AI
            
        iteration_results[iteration] = model.objVal
                    
        
        
    result = {
        "status": "Optimal" if model.status == GRB.OPTIMAL else f"Status {model.status}",
        "n_nodes": n_nodes,
        "objective_value": model.objVal if model.status == GRB.OPTIMAL else None,
        "tour": [],
        "runtime": model.Runtime,
        "iteration_results": iteration_results
    }
    

    return result
    
def test_dfj_improve(df_nodes, instances, time_limit):
    
    dfj_naive_results = []

    for n in instances:
        
        res_dfj = solve_tsp_dfj_improve(df_nodes, n, time_limit)
        dfj_time = res_dfj['runtime']
        dfj_obj = res_dfj['objective_value'] if res_dfj['objective_value'] != "N/A" else "N/A"
        dfj_status = res_dfj['status']
        
        
        dfj_naive_results.append({
            "Customers": n,
            "Formulation": "DFJ Improve",
            "Status": dfj_status,
            "Time_s": round(dfj_time, 2),
            "Objective": dfj_obj
        })
    
    df_dfj_results = pd.DataFrame(dfj_naive_results)
    
    os.makedirs(r"Results\DFJ", exist_ok=True)
    df_dfj_results.to_csv(r"Results\DFJ\dfj_improve_results.csv", index=False)
    

    
    return df_dfj_results    

def plot_dfj_iterations(iteration_results):
    
   
    
    iterations = list(iteration_results.keys())
    objectives = list(iteration_results.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, objectives, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Objective Value (Total Distance)', fontsize=12)
    plt.title('DFJ Improved: Objective Value Evolution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(iterations)
    plt.tight_layout()
    os.makedirs('Results/DFJ', exist_ok=True)
    plt.savefig('Results/DFJ/dfj_iterations.png', dpi=300, bbox_inches='tight')
    plt.close()

def solve_vrptw_MTZ(df_nodes, df_requests, df_Fleet, n_customers, time_limit=600, vehicle_cost=0, driver_duration= False):
    """
    Solves VRPTW using MTZ formulation with capacity and time window constraints.
    
    """
    
    nodes = df_nodes[df_nodes['id'] <= n_customers].copy()
    
    coords = nodes[['cx', 'cy']].values
    n_nodes = len(coords)
    
    
    dist_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                dist_matrix[i][j] = round(np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2))
    
    capacity = df_Fleet.loc[0, 'capacity']
    
    ids = nodes['id'].values
    demands = np.zeros(n_nodes)
    service_times = np.zeros(n_nodes)
    early_start = np.zeros(n_nodes)
    late_start = np.zeros(n_nodes)
    
    for i in range(1, n_nodes):
        node_id = ids[i]
        req = df_requests[df_requests['id'] == node_id].iloc[0]
        demands[i] = req['quantity']
        service_times[i] = req['service_time']
        early_start[i] = req['start']
        late_start[i] = req['end']
    
    max_time = float(df_Fleet['max_travel_time'].iloc[0])
    

    
    model = gp.Model("VRPTW_MTZ")
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)
    
    x = model.addVars(n_nodes, n_nodes, vtype=GRB.BINARY, name="x")
    
    u = model.addVars(n_nodes, lb=0.0, ub=capacity, vtype=GRB.CONTINUOUS, name="u") # load variable
    
    early_start[0] = 0
    late_start[0] = max_time
    
    t = model.addVars(n_nodes, lb=0.0, ub=max_time, vtype=GRB.CONTINUOUS, name="t") # time variable
    
    
    model.addConstrs((gp.quicksum(x[i, j] for j in range(n_nodes) if j != i) == 1 
                      for i in range(1, n_nodes)), name="Leave_Customer")
    

    model.addConstrs((gp.quicksum(x[i, j] for i in range(n_nodes) if i != j) == 1 
                      for j in range(1, n_nodes)), name="Enter_Customer")
    
    model.addConstr(
        gp.quicksum(x[0, j] for j in range(1, n_nodes)) == gp.quicksum(x[i, 0] for i in range(1, n_nodes)), name="Depot")
    
    BigM = 1e5
    

    for i in range(n_nodes): # load
        for j in range(1, n_nodes): 
            if i != j:
                model.addConstr(
                    u[j] >= u[i] + demands[j] - BigM * (1 - x[i, j]),
                )
    
    # depot load zero
    model.addConstr(u[0] == 0, name="Depot_Load_Zero")
    
    
    for i in range(1, n_nodes):
        model.addConstr(u[i] >= demands[i])
        model.addConstr(u[i] <= capacity)
    
    for i in range(n_nodes):
        model.addConstr(t[i] >= early_start[i])
        model.addConstr(t[i] <= late_start[i])
    
   
    

    for i in range(n_nodes):
        for j in range(1, n_nodes):  
            if i != j:
                model.addConstr(
                    t[i] + service_times[i] + dist_matrix[i][j] - t[j] <= BigM * (1 - x[i, j]),
                    
                )
    
    if driver_duration:
        rt_start = model.addVars(n_nodes, lb=0.0, ub=max_time, vtype=GRB.CONTINUOUS, name="rt_start")
        for j in range(1, n_nodes):
            model.addConstr(rt_start[j] <= t[j] - dist_matrix[0][j] + BigM * (1 - x[0, j]), f"StartInit_{j}")
        for i in range(1, n_nodes):
            for j in range(1, n_nodes):
                if i != j:
                    model.addConstr(rt_start[j] >= rt_start[i] - BigM * (1 - x[i, j]), f"PropStart_LB_{i}_{j}")
                    model.addConstr(rt_start[j] <= rt_start[i] + BigM * (1 - x[i, j]), f"PropStart_UB_{i}_{j}")
        
        for i in range(1, n_nodes):
            # If x[i,0]=1 (vehicle returns to depot from i):
            # Finish Time = t[i] + service[i] + dist[i,0]
            # Duration = Finish Time - rt_start[i]
            finish_time = t[i] + service_times[i] + dist_matrix[i][0]
            model.addConstr(finish_time - rt_start[i] <= 420 + BigM * (1 - x[i, 0]), f"Duration_{i}")
    

    num_vehicles = gp.quicksum(x[0, j] for j in range(1, n_nodes))
    total_distance = gp.quicksum(dist_matrix[i][j] * x[i, j] for i in range(n_nodes) for j in range(n_nodes) if i != j)
    
    model.setObjective(total_distance + vehicle_cost * num_vehicles, GRB.MINIMIZE)

    model.optimize()
    
    result = {
        "status": None,
        "n_nodes": n_nodes,
        "objective_value": None,
        "tour": [],
        "runtime": model.Runtime
    }
    
    
    # this is AI helped, for getting number of vehicles and routes
    if model.status == GRB.OPTIMAL:
        result["status"] = "Optimal"
        result["objective_value"] = model.objVal

        routes = []
        
        arcs = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j and x[i, j].X > 0.5]
        
        for j in range(1, n_nodes):
            if x[0, j].X > 0.5:  
                route = [0, j]  
                current = j
                
                while True:
                    # Find next node in route
                    next_node = None
                    for i, k in arcs:
                        if i == current and k != 0:
                            next_node = k
                            break
                        elif i == current and k == 0:
                            next_node = 0
                            break
                    
                    if next_node is None:
                        break
                    
                    route.append(next_node)
                    
                    if next_node == 0:  
                        break
                    
                    current = next_node
                
                routes.append(route)
        
        result["routes"] = routes
        result["num_vehicles"] = len(routes)
        
        # Calculate actual distance traveled (excluding vehicle cost)
        total_distance = sum(dist_matrix[route[i]][route[i+1]] for route in routes for i in range(len(route)-1))
        result["total_distance"] = total_distance
        
        # Create unique plot based on parameters
        duration_suffix = "_duration" if driver_duration else ""
        plot_vrptw_routes(nodes, routes, coords, n_customers, vehicle_cost, driver_duration)
        


    
    
    return result

def solve_vrptw_TimeExpanded(df_nodes, df_requests, df_Fleet, n_customers, delta=1, time_limit=600):

    start_time = time.time()
    
    nodes = df_nodes[df_nodes['id'] <= n_customers].copy()
    coords = nodes[['cx', 'cy']].values
    n_nodes = len(coords)
    
    dist_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                dist_matrix[i][j] = round(np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2))
    
    capacity = float(df_Fleet.loc[0, 'capacity'])
    max_travel_time = float(df_Fleet.loc[0, 'max_travel_time'])
    
    ids = nodes['id'].values
    demands = np.zeros(n_nodes)
    service_times = np.zeros(n_nodes)
    early_start = np.zeros(n_nodes)
    late_start = np.zeros(n_nodes)
    
    for i in range(1, n_nodes):
        node_id = ids[i]
        req = df_requests[df_requests['id'] == node_id].iloc[0]
        demands[i] = req['quantity']
        service_times[i] = req['service_time']
        early_start[i] = req['start']
        late_start[i] = req['end']
        
    # Depot time window
    early_start[0] = 0
    late_start[0] = max_travel_time


    node_time_map = {}  
    # feasible time points for each node
    for i in range(n_nodes):
        node_time_map[i] = []
        start_t = int(np.ceil(early_start[i] / delta) * delta)
        end_t = int(np.floor(late_start[i] / delta) * delta)
        for t in range(start_t, end_t + 1, delta):
            node_time_map[i].append(t)
    
            
    arcs = []
    arc_costs = {}
    # arcs: (i, t, j, t') from node i at time t to node j at time t'
    
    for i in range(n_nodes):
        for t in node_time_map[i]:
            for j in range(n_nodes):
                if i == j: continue
                
                arrival_at_j = t + service_times[i] + dist_matrix[i][j] # speed is 1 per unit distance
                min_t_prime = int(np.ceil(arrival_at_j / delta) * delta)
                # print(arrival_at_j)
                for t_prime in node_time_map[j]:
                    if t_prime >= min_t_prime:
                        arc = (i, t, j, t_prime)
                        arcs.append(arc)
                        arc_costs[arc] = dist_matrix[i][j]


    model = gp.Model("VRPTW_TimeExpanded")
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)
    
    
    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    
    # y[i,t] = 1 if customer i is serviced starting at time t
    y_keys = [(i,t) for i in range(1, n_nodes) for t in node_time_map[i]]
    y = model.addVars(y_keys, vtype=GRB.BINARY, name="y") # [1,100] , [1,105] ...   
    
    u = model.addVars(n_nodes, lb=0.0, ub=capacity, vtype=GRB.CONTINUOUS, name="u")
    
    
    # One Visit
    for i in range(1, n_nodes):
        model.addConstr(gp.quicksum(y[i, t] for t in node_time_map[i]) == 1)
        
    # in and out
    for i in range(1, n_nodes):
        for t in node_time_map[i]:
            outgoing_arcs = [a for a in arcs if a[0]==i and a[1]==t] # all arcs starting from (i,t)
            if not outgoing_arcs:
                model.addConstr(y[i,t] == 0) # If vehicle cannot leave, then y[i,t]=0 it is too late
            else:
                model.addConstr(gp.quicksum(x[a] for a in outgoing_arcs) == y[i,t])
                
   
    for j in range(1, n_nodes):
        for t_prime in node_time_map[j]:
            incoming_arcs = [a for a in arcs if a[2]==j and a[3]==t_prime] # all arcs arriving at (j,t')
            model.addConstr(gp.quicksum(x[a] for a in incoming_arcs) == y[j,t_prime])

    depot_outgoing = [a for a in arcs if a[0]==0] # arcs leaving depot
    depot_incoming = [a for a in arcs if a[2]==0] # arcs arriving at depot
    model.addConstr(gp.quicksum(x[a] for a in depot_outgoing) == gp.quicksum(x[a] for a in depot_incoming))
    
    
    BigM = 1e5
    
    for a in arcs:
        i, t, j, t_prime = a
        if i != 0 and j != 0: 
            model.addConstr(u[j] >= u[i] + demands[j] - BigM * (1 - x[a]))
            
            
    model.addConstr(u[0] == 0)
    for i in range(1, n_nodes):
        model.addConstr(u[i] >= demands[i])
        model.addConstr(u[i] <= capacity)



    model.setObjective(gp.quicksum(arc_costs[a] * x[a] for a in arcs) , GRB.MINIMIZE)
    
    model.optimize()
    
    result = {
        "status": "Infeasible",
        "objective_value": None,
        "total_distance": None,
        "num_vehicles": None,
        "runtime": time.time() - start_time,
        "routes": []
    }
    
    if model.status == GRB.OPTIMAL:
        result["status"] = "Optimal"
        result["objective_value"] = model.objVal
        result["total_distance"] = sum(arc_costs[a] * x[a].X for a in arcs if x[a].X > 0.5)
        result["num_vehicles"] = sum(x[a].X for a in depot_outgoing if x[a].X > 0.5)
        
    elif model.status == GRB.TIME_LIMIT:
        result["status"] = "Time Limit"
        if model.SolCount > 0:
            result["objective_value"] = model.objVal
            result["total_distance"] = sum(arc_costs[a] * x[a].X for a in arcs if x[a].X > 0.5)
            result["num_vehicles"] = sum(x[a].X for a in depot_outgoing if x[a].X > 0.5)
            
    return result

def test_compare_formulations(df_nodes, df_requests, df_Fleet, instances, time_limit):

    results = []
    
    for n in instances:
        for delta in [1, 5, 10, 20]:
            res_te = solve_vrptw_TimeExpanded(df_nodes, df_requests, df_Fleet, n, delta=delta, time_limit=time_limit)

            results.append({
                'Customers': n,
                'Formulation': 'Time-Expanded',
                'Delta': delta,
                'Status': res_te['status'],
                'Objective': res_te['objective_value'] if res_te['objective_value'] is not None else 'N/A',
                'Distance': res_te.get('total_distance', 'N/A'),
                'Runtime_s': round(res_te['runtime'], 2)
            })
    
    df_comparison = pd.DataFrame(results)
    os.makedirs(r"Results\TimeExpanded", exist_ok=True)
    df_comparison.to_csv(r"Results\TimeExpanded\mtz_vs_timeexpanded.csv", index=False)
    

    
    return df_comparison

def test_vrptw_mtz(df_nodes, df_requests, df_Fleet, instances, time_limit, vehicle_cost=0):
    
    vrptw_results = []
    
    for n in instances:
        
        result = solve_vrptw_MTZ(df_nodes, df_requests, df_Fleet, n_customers=n, time_limit=time_limit, vehicle_cost=vehicle_cost)
        
        vrptw_results.append({
            "Customers": n,
            "Formulation": "VRPTW_MTZ",
            "Status": result['status'],
            "Time_s": round(result['runtime'], 2),
            "Objective": result['objective_value'] if result['objective_value'] is not None else "N/A",
            "Distance": result.get('total_distance', "N/A"),
            "Num_Vehicles": result.get('num_vehicles', "N/A"),
            "Vehicle_Cost": vehicle_cost
        })
    
    df_vrptw_results = pd.DataFrame(vrptw_results)
    
    os.makedirs(r"Results\VRPTW", exist_ok=True)
    df_vrptw_results.to_csv(r"Results\VRPTW\vrptw_mtz_results.csv", index=False)
    
    return df_vrptw_results

def analyze_vehicle_tradeoff(df_nodes, df_requests, df_Fleet, n_customers, vehicle_costs, time_limit=600):

    results = []

    
    for cost in vehicle_costs:
        result = solve_vrptw_MTZ(df_nodes, df_requests, df_Fleet, n_customers=n_customers, 
                                 time_limit=time_limit, vehicle_cost=cost)
        
        if result['status'] == 'Optimal':
            results.append({
                'Vehicle_Cost': cost,
                'Num_Vehicles': result['num_vehicles'],
                'Total_Distance': result['total_distance'],
                'Objective_Value': result['objective_value'],
                'Runtime_s': round(result['runtime'], 2)
            })
            print(f"  → Vehicles: {result['num_vehicles']}, Distance: {result['total_distance']:.2f}\n")
    
    df_tradeoff = pd.DataFrame(results)
    os.makedirs(r"Results\VehicleTradeoff", exist_ok=True)
    df_tradeoff.to_csv(rf"Results\VehicleTradeoff\vehicle_tradeoff_n{n_customers}.csv", index=False)
    
    
    
    return df_tradeoff

def modify_time_windows(df_requests, factor):
    """
    Modifies time windows by scaling their width around the center point.
    
    Args:
        df_requests: Original requests DataFrame
        factor: Scaling factor (0.5 = half width, 1.0 = original, 2.0 = double width)
    
    Returns:
        Modified DataFrame with scaled time windows
    """
    df_modified = df_requests.copy()
    
    for id in df_modified.index:
        original_start = df_requests.loc[id, 'start']
        original_end = df_requests.loc[id, 'end']
        
        center = (original_start + original_end) / 2
        half_width = (original_end - original_start) / 2
        
        new_half_width = half_width * factor
        df_modified.loc[id, 'start'] = int(max(0, center - new_half_width))
        df_modified.loc[id, 'end'] = int(center + new_half_width)
    
    return df_modified

def analyze_timewindow_tradeoff(df_nodes, df_requests, df_Fleet, n_customers, window_factors, time_limit=600):

    results = []
    
    
    for factor in window_factors:
        
        df_requests_modified = modify_time_windows(df_requests, factor)
        
        result = solve_vrptw_MTZ(df_nodes, df_requests_modified, df_Fleet,
                                n_customers=n_customers,
                                time_limit=time_limit,
                                vehicle_cost=0)
        
        if result['status'] == 'Optimal':
            results.append({
                'Window_Factor': factor,
                'Num_Vehicles': result['num_vehicles'],
                'Total_Distance': result['total_distance'],
                'Objective_Value': result['objective_value'],
                'Runtime_s': round(result['runtime'], 2),
                'Status': 'Feasible'
            })
        else:
            results.append({
                'Window_Factor': factor,
                'Num_Vehicles': 'N/A',
                'Total_Distance': 'N/A',
                'Objective_Value': 'N/A',
                'Runtime_s': round(result['runtime'], 2),
                'Status': 'Infeasible'
            })
    
    df_tradeoff = pd.DataFrame(results)
    plot_timewindow_tradeoff(df_tradeoff, n_customers)
    
    return df_tradeoff

def plot_timewindow_tradeoff(df_tradeoff, n_customers):

    

    plt.figure(figsize=(10, 7))
    
    plt.plot(df_tradeoff['Window_Factor'], df_tradeoff['Total_Distance'],
             'bo-', linewidth=2.5, markersize=12, label='Total Distance')
    plt.xlabel('Time Window Factor (1.0 = Original Width)', fontsize=13)
    plt.ylabel('Total Distance', fontsize=13)
    plt.title(f'Trade-off: Distance vs Time Window Width (n={n_customers})', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Original')
    plt.legend(fontsize=11)
    

    
    plt.tight_layout()
    os.makedirs('Results/Plots_timeWindow', exist_ok=True)
    plt.savefig(f'Results/Plots_timeWindow/timewindow_tradeoff_n{n_customers}.png', dpi=300, bbox_inches='tight')
    plt.close()
    

    
   
    
    plt.figure(figsize=(12, 8))
    
    # Plot the Pareto frontier
    plt.plot(df_tradeoff['Num_Vehicles'], df_tradeoff['Total_Distance'], 
             'go-', linewidth=3, markersize=14, label='Pareto Solutions', markeredgecolor='darkgreen', markeredgewidth=2)
    
    plt.xlabel('Number of Vehicles', fontsize=14, fontweight='bold')
    plt.ylabel('Total Distance (km)', fontsize=14, fontweight='bold')
    plt.title(f'Trade-off: Vehicles vs Distance (n={n_customers})', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.legend(fontsize=12, loc='best')
    
    # Add cost annotations with better positioning
    for idx, row in df_tradeoff.iterrows():
        plt.annotate(f"cost={int(row['Vehicle_Cost'])}", 
                    xy=(row['Num_Vehicles'], row['Total_Distance']),
                    xytext=(10, -15) if idx % 2 == 0 else (10, 15), 
                    textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5))
    
    # Set integer ticks for number of vehicles
    vehicles = df_tradeoff['Num_Vehicles'].values
    plt.xticks(range(int(vehicles.min()), int(vehicles.max()) + 1))
    
    # Add margin to axes
    x_margin = 0.5
    y_range = df_tradeoff['Total_Distance'].max() - df_tradeoff['Total_Distance'].min()
    y_margin = y_range * 0.1 if y_range > 0 else 5
    
    plt.xlim(vehicles.min() - x_margin, vehicles.max() + x_margin)
    plt.ylim(df_tradeoff['Total_Distance'].min() - y_margin, 
             df_tradeoff['Total_Distance'].max() + y_margin)
    
    plt.tight_layout()
    os.makedirs(r"Results\VehicleTradeoff", exist_ok=True)
    plt.savefig(f'Results/VehicleTradeoff/vehicle_tradeoff_n{n_customers}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_vrptw_routes(nodes, routes, coords, n_customers, vehicle_cost, driver_duration=False):
   
    
    
    plt.figure(figsize=(12, 10))
    
    # Define colors for different vehicles
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot depot
    plt.scatter(coords[0, 0], coords[0, 1], c='black', s=300, marker='s', 
                label='Depot', zorder=5, edgecolors='white', linewidths=2)
    plt.text(coords[0, 0], coords[0, 1], '0', fontsize=10, ha='center', va='center', 
             color='white', fontweight='bold')
    
    # Plot each route
    for idx, route in enumerate(routes):
        color = colors[idx % len(colors)]
        
        # Plot route edges
        for i in range(len(route) - 1):
            node_from = route[i]
            node_to = route[i + 1]
            
            x_coords = [coords[node_from, 0], coords[node_to, 0]]
            y_coords = [coords[node_from, 1], coords[node_to, 1]]
            
            plt.plot(x_coords, y_coords, c=color, linewidth=2, alpha=0.7, zorder=1)
            
            # Add arrow
            dx = x_coords[1] - x_coords[0]
            dy = y_coords[1] - y_coords[0]
            plt.arrow(x_coords[0] + dx*0.5, y_coords[0] + dy*0.5, 
                     dx*0.15, dy*0.15, head_width=0.5, head_length=0.3, 
                     fc=color, ec=color, alpha=0.6, zorder=2)
        
        # Plot customer nodes for this route (excluding depot)
        route_customers = [node for node in route if node != 0]
        if route_customers:
            route_coords = coords[route_customers]
            plt.scatter(route_coords[:, 0], route_coords[:, 1], 
                       c=color, s=150, alpha=0.8, edgecolors='black', 
                       linewidths=1.5, zorder=3, label=f'Vehicle {idx+1}')
            
            # Add node labels
            for node in route_customers:
                plt.text(coords[node, 0], coords[node, 1], str(node), 
                        fontsize=8, ha='center', va='center', fontweight='bold')
    
    # Create title based on configuration
    title = f'VRPTW Solution - Vehicle Routes (n={n_customers}'
    if driver_duration:
        title += ', 7h limit'
    if vehicle_cost > 0:
        title += f', cost={vehicle_cost}'
    title += ')'

    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    # Create descriptive filename
    duration_suffix = "_duration7h" if driver_duration else ""
    cost_suffix = f"_cost{vehicle_cost}" if vehicle_cost > 0 else ""
    
    os.makedirs('Results/VRPTW_Plots', exist_ok=True)
    plt.savefig(f'Results/VRPTW_Plots/vrptw_n{n_customers}{duration_suffix}{cost_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_driver_duration(df_nodes, df_requests, df_Fleet, instances, time_limit=600):
    
    
    results = []


    for n in instances:
        print(f"\nProcessing Instance n={n}...")
        
    
        res_unlimited = solve_vrptw_MTZ(
            df_nodes, df_requests, df_Fleet, 
            n_customers=n, 
            time_limit=time_limit,
            vehicle_cost=0,
            driver_duration=False
        )
        

        res_limited = solve_vrptw_MTZ(
            df_nodes, df_requests, df_Fleet, 
            n_customers=n, 
            time_limit=time_limit,
            vehicle_cost=0,
            driver_duration=True
        )
        

        results.append({
            "Customers": n,
            "Dist_Unlim": round(res_unlimited['objective_value'], 2) if res_unlimited['objective_value'] is not None else "N/A",
            "Veh_Unlim": res_unlimited.get('num_vehicles', "N/A"),
            "Dist_7h": round(res_limited['objective_value'], 2) if res_limited['objective_value'] is not None else "N/A",
            "Veh_7h": res_limited.get('num_vehicles', "N/A"),
            "Status_7h": res_limited['status']
        })

    df_results = pd.DataFrame(results)
    
    os.makedirs(r"Results\DriverDuration", exist_ok=True)
    df_results.to_csv(r"Results\DriverDuration\driver_duration_analysis.csv", index=False)
    

    
    return df_results

def test_vehicle_tradeoff(df_nodes, df_requests, df_Fleet, instances, vehicle_costs, time_limit=600):

    
    for n in instances:
    
        analyze_vehicle_tradeoff(df_nodes, df_requests, df_Fleet, 
                                               n_customers=n, 
                                               vehicle_costs=vehicle_costs,
                                               time_limit=time_limit)
    
    print("\nVehicle trade-off analysis complete!")
    
    return

def analyze_timewindow_tradeoff(df_nodes, df_requests, df_Fleet, n_customers, window_factors, time_limit=600):
    
    results = []
    
    for factor in window_factors:
        
        df_requests_modified = modify_time_windows(df_requests, factor)
        
        result = solve_vrptw_MTZ(df_nodes, df_requests_modified, df_Fleet,
                                n_customers=n_customers,
                                time_limit=time_limit,
                                vehicle_cost=0)
        
        results.append({
            'Window_Factor': factor,
            'Num_Vehicles': result['num_vehicles'],
            'Total_Distance': result['total_distance'],
            'Objective_Value': result['objective_value'],
                'Runtime_s': round(result['runtime'], 2),
                'Status': 'Feasible'
            })

    df_tradeoff = pd.DataFrame(results)
    os.makedirs(r"Results\timewindow_tradeoff", exist_ok=True)
    df_tradeoff.to_csv(rf"Results\timewindow_tradeoff\timewindow_tradeoff_n{n_customers}.csv", index=False)
    plot_timewindow_tradeoff(df_tradeoff, n_customers)
    
    
    return df_tradeoff

if __name__ == "__main__":
    print("Loading data...")
    fname = "customers.xlsx"
    df_nodes = pd.read_excel(fname, sheet_name=0, engine='openpyxl')
    df_requests = pd.read_excel(fname, sheet_name=1, engine='openpyxl')
    df_Fleet = pd.read_excel(fname, sheet_name=2, engine='openpyxl')
    

 
    #test_mtz(df_nodes, [5, 10, 15, 20, 25], time_limit=600)
    
    
    #test_dfj(df_nodes, [5, 10, 15, 20, 25], time_limit=600)
    
    
    #test_dfj_improve(df_nodes, [5, 10, 15, 20, 25], time_limit=600)
    #plot_dfj_iterations(solve_tsp_dfj_improve(df_nodes, 25, 600)['iteration_results'])
    
    
    #test_driver_duration(df_nodes, df_requests, df_Fleet, instances=[5, 10, 15, 20, 25], time_limit=600)  
    
   # for n in [5,10, 15, 20,25]:
    #    analyze_timewindow_tradeoff(df_nodes, df_requests, df_Fleet, n_customers=n, window_factors=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0], time_limit=600)
    
    #test_vrptw_mtz(df_nodes, df_requests, df_Fleet, instances=[5, 10, 15, 20, 25], time_limit=600, vehicle_cost=0)
    #test_vehicle_tradeoff(df_nodes, df_requests, df_Fleet, instances=[5, 10, 15, 20, 25],vehicle_costs=[-200, -500, -1000, 20, 50, 200, 300, 500,2000000],time_limit=600)
    
    
    #test_compare_formulations(df_nodes, df_requests, df_Fleet, instances=[5, 10, 15, 20, 25], time_limit=300)

    

    



