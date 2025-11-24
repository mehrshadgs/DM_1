import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import math

#parameters
timelimit = 600
instance_sizes = [5]

#load data
nodes_df = pd.read_excel("customers.xlsx",sheet_name="Nodes")
request_df = pd.read_excel("customers.xlsx",sheet_name="Requests")
fleet_df = pd.read_excel("customers.xlsx",sheet_name="Fleet")  

depot = nodes_df[nodes_df['type'] == 0]
depot_id = int(depot['id'].iloc[0])

customer_id = nodes_df[nodes_df['type'] == 1]['id'].tolist()

all_nodes = nodes_df['id'].tolist()
coordinates = nodes_df.set_index("id")[["cx","cy"]].to_dict("index")

# Euclidean distance 
def dist(i, j):
    x = coordinates[i]["cx"] - coordinates[j]["cx"]
    y = coordinates[i]["cy"] - coordinates[j]["cy"]
    distance = int(round(math.sqrt(x**2 + y**2)))
    return distance

for n in instance_sizes:

    node_ids = [depot_id] + customer_id[:n]
    n_nr = len(node_ids)

    MTZ_model = gp.Model("MTZ_TSP")
    MTZ_model.setParam('TimeLimit', timelimit)
    MTZ_model.setParam('OutputFlag', 1)

    x = MTZ_model.addVars(n_nr, n_nr, vtype=GRB.BINARY, name="x")

    for i in range(n_nr):
        x[i, i].ub = 0
    
    u = MTZ_model.addVars(n_nr, vtype=GRB.CONTINUOUS, lb=0, ub=n_nr-1, name="u")

    # objective
    MTZ_model.setObjective(gp.quicksum(dist(node_ids[i], node_ids[j]) * x[i,j]
                                   for i in range(n_nr) for j in range(n_nr)),
                       GRB.MINIMIZE)

    #constraints
    for i in range(n_nr):
        MTZ_model.addConstr(gp.quicksum(x[i,j] for j in range(n_nr) if j != i) == 1)
        MTZ_model.addConstr(gp.quicksum(x[j,i] for j in range(n_nr) if j != i) == 1)

    MTZ_model.addConstr(u[0] == 0)  
    for i in range(1, n_nr):
        MTZ_model.addConstr(u[i] >= 1)
        MTZ_model.addConstr(u[i] <= n_nr - 1)

    for i in range(1, n_nr):
        for j in range(1, n_nr):
            if i != j:
                MTZ_model.addConstr(u[i] - u[j] + n_nr * x[i,j] <= n_nr - 1)

    MTZ_model.optimize()

    if MTZ_model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        sol = MTZ_model.getAttr("x", x)

        tour = [0]   # start at depot
        current = 0

        for a in range(n_nr - 1):
            for j in range(n_nr):
                if sol[current, j] > 0.5:
                    tour.append(j)
                    current = j
                    break
        tour_ids = [node_ids[i] for i in tour]
        tour_ids.append(node_ids[0])  

        print("Objective =", MTZ_model.objVal)
        print("Tour =", tour_ids)

    
