def solve_vrptw_TimeExpanded(df_nodes, df_requests, df_Fleet, n_customers, delta=1, time_limit=600, vehicle_cost=0):
    """
    Solves VRPTW using a Hybrid Time-Expanded Network formulation.
    Uses time-expanded nodes for temporal feasibility and MTZ variables for capacity.
    """
    import gurobipy as gp
    from gurobipy import GRB
    import numpy as np
    import time

    start_time = time.time()
    
    # 1. Prepare Data
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

    # 2. Build Time-Expanded Graph
    # Discretize time
    time_horizon = int(max_travel_time)
    
    # Pre-calculate feasible time points for each node to reduce graph size
    # For a node i, feasible t are in [early_start[i], late_start[i]] with step delta
    feasible_nodes = [] # List of tuples (i, t)
    node_time_map = {}  # Map i -> list of t
    
    for i in range(n_nodes):
        node_time_map[i] = []
        # Calculate valid range (align with delta)
        start_t = int(np.ceil(early_start[i] / delta) * delta)
        end_t = int(np.floor(late_start[i] / delta) * delta)
        
        for t in range(start_t, end_t + 1, delta):
            feasible_nodes.append((i, t))
            node_time_map[i].append(t)
            
    # Create Arcs: ((i,t), (j,t'))
    arcs = []
    arc_costs = {}
    
    # To optimize capacity constraints later, we track which time-arcs belong to physical arc (i,j)
    physical_arc_map = {} # (i,j) -> list of time-expanded arcs
    
    for i in range(n_nodes):
        for t in node_time_map[i]:
            for j in range(n_nodes):
                if i == j: continue
                
                # Travel time + Service time
                # Note: t is arrival/start service at i. We finish service at i at t + service_times[i]
                # Then travel to j. Arrival at j is t + service_times[i] + dist[i][j]
                
                arrival_at_j = t + service_times[i] + dist_matrix[i][j]
                
                # Check feasibility at j
                # We can arrive at j at 'arrival_at_j', but service can start any time t' >= arrival_at_j
                # subject to t' <= late_start[j].
                # In strict time-expanded graphs, we usually have "waiting arcs" (j,t)->(j,t+1).
                # To simplify, we allow direct arcs to any valid future time t' at j.
                
                min_t_prime = int(np.ceil(arrival_at_j / delta) * delta)
                
                # optimization: Only look at feasible t' for j
                for t_prime in node_time_map[j]:
                    if t_prime >= min_t_prime:
                        # This arc is valid
                        arc = (i, t, j, t_prime)
                        arcs.append(arc)
                        arc_costs[arc] = dist_matrix[i][j]
                        
                        if (i,j) not in physical_arc_map:
                            physical_arc_map[(i,j)] = []
                        physical_arc_map[(i,j)].append(arc)

    print(f"Graph built: {len(feasible_nodes)} time-nodes, {len(arcs)} time-arcs")

    # 3. Model
    model = gp.Model("VRPTW_TimeExpanded")
    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 1) # Set to 1 to see progress if needed

    # Variables
    # x[i,t,j,t'] = 1 if vehicle goes from i (serviced at t) to j (serviced at t')
    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    
    # y[i,t] = 1 if customer i is serviced starting at time t
    # (We only need this for customers 1..N, depot is special)
    y_keys = [(i,t) for i in range(1, n_nodes) for t in node_time_map[i]]
    y = model.addVars(y_keys, vtype=GRB.BINARY, name="y")
    
    # u[i] = Cumulative load variable (Standard MTZ)
    u = model.addVars(n_nodes, lb=0.0, ub=capacity, vtype=GRB.CONTINUOUS, name="u")
    
    # Constraints
    
    # 1. Visit each customer exactly once
    for i in range(1, n_nodes):
        model.addConstr(gp.quicksum(y[i, t] for t in node_time_map[i]) == 1, name=f"Visit_{i}")
        
    # 2. Link Flow to Node Visit (Out)
    for i in range(1, n_nodes):
        for t in node_time_map[i]:
            # Sum of outgoing arcs from (i,t) must equal y[i,t]
            # outgoing = arcs starting with i,t
            outgoing_arcs = [a for a in arcs if a[0]==i and a[1]==t]
            if not outgoing_arcs:
                model.addConstr(y[i,t] == 0) # Dead end in time
            else:
                model.addConstr(gp.quicksum(x[a] for a in outgoing_arcs) == y[i,t], name=f"FlowOut_{i}_{t}")

    # 3. Link Flow to Node Visit (In)
    for j in range(1, n_nodes):
        for t_prime in node_time_map[j]:
            # Sum of incoming arcs to (j,t') must equal y[j,t']
            incoming_arcs = [a for a in arcs if a[2]==j and a[3]==t_prime]
            model.addConstr(gp.quicksum(x[a] for a in incoming_arcs) == y[j,t_prime], name=f"FlowIn_{j}_{t_prime}")

    # 4. Depot Flow Balance
    # Sum of all arcs leaving depot = Sum of all arcs entering depot ( = Num Vehicles)
    depot_outgoing = [a for a in arcs if a[0]==0]
    depot_incoming = [a for a in arcs if a[2]==0]
    
    model.addConstr(gp.quicksum(x[a] for a in depot_outgoing) == gp.quicksum(x[a] for a in depot_incoming), name="DepotBalance")
    
    # 5. Capacity Constraints (MTZ style on aggregate flow)
    # We aggregate flow on physical edge (i,j)
    BigM = capacity
    
    # Only need to iterate over physical edges that actually exist in time-expanded graph
    for (i, j), time_arcs in physical_arc_map.items():
        if i != 0 and j != 0: # Standard MTZ for customers
            # X_ij = sum of all time-expanded arcs for this physical link
            X_ij = gp.quicksum(x[a] for a in time_arcs)
            model.addConstr(u[j] >= u[i] + demands[j] - BigM * (1 - X_ij), name=f"Cap_{i}_{j}")
            
    # Depot load and boundaries
    model.addConstr(u[0] == 0)
    for i in range(1, n_nodes):
        model.addConstr(u[i] >= demands[i])
        model.addConstr(u[i] <= capacity)

    # Objective
    obj_expr = gp.quicksum(arc_costs[a] * x[a] for a in arcs)
    
    # Add vehicle cost if specified
    num_vehicles = gp.quicksum(x[a] for a in depot_outgoing)
    model.setObjective(obj_expr + vehicle_cost * num_vehicles, GRB.MINIMIZE)
    
    # Solve
    model.optimize()
    
    result = {
        "status": "Infeasible",
        "objective_value": None,
        "total_distance": None,
        "num_vehicles": None,
        "runtime": time.time() - start_time
    }
    
    if model.status == GRB.OPTIMAL:
        result["status"] = "Optimal"
        result["objective_value"] = model.objVal
        result["total_distance"] = sum(arc_costs[a] * x[a].X for a in arcs if x[a].X > 0.5)
        result["num_vehicles"] = sum(x[a].X for a in depot_outgoing if x[a].X > 0.5)
        
    return result

def test_compare_formulations(df_nodes, df_requests, df_Fleet, n_customers=15):
    """
    Compares MTZ vs Time-Expanded formulations
    """
    print(f"\n--- Comparing Formulations (n={n_customers}) ---")
    
    # 1. MTZ
    print("Running MTZ...")
    res_mtz = solve_vrptw_MTZ(df_nodes, df_requests, df_Fleet, n_customers, time_limit=300)
    print(f"MTZ: Obj={res_mtz['objective_value']}, Time={res_mtz['runtime']:.2f}s")
    
    # 2. Time Expanded (Delta=1 for exact, Delta=10 for approx)
    print("Running Time-Expanded (Delta=20)...")
    res_te = solve_vrptw_TimeExpanded(df_nodes, df_requests, df_Fleet, n_customers, delta=20, time_limit=300)
    print(f"TE (d=20): Obj={res_te['objective_value']}, Time={res_te['runtime']:.2f}s")

    # Optional: Run with Delta=1 if n is small
    if n_customers <= 10:
        print("Running Time-Expanded (Delta=1)...")
        res_te_exact = solve_vrptw_TimeExpanded(df_nodes, df_requests, df_Fleet, n_customers, delta=1, time_limit=300)
        print(f"TE (d=1): Obj={res_te_exact['objective_value']}, Time={res_te_exact['runtime']:.2f}s")