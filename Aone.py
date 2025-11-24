import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
from itertools import combinations
import matplotlib.pyplot as plt



def solve_tsp_mtz(df_nodes, n_customers, time_limit):
    """
    Solves the Traveling Salesperson Problem (TSP) using the MTZ formulation.
    
    """
    
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
    df_mtz_results.to_csv("mtz_results.csv", index=False)
    
    
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
    df_dfj_results.to_csv("dfj_results.csv", index=False)
    

    
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
    df_dfj_results.to_csv("dfj_improve_results.csv", index=False)
    

    
    return df_dfj_results    

def plot_dfj_iterations(iteration_results):
    iterations = list(iteration_results.keys())
    objectives = list(iteration_results.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, objectives, marker='o')
    plt.title('DFJ Improvement Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.grid(True)
    plt.show()

fname = "customers.xlsx" 
df_nodes = pd.read_excel(fname, sheet_name=0, engine='openpyxl')
df_requests = pd.read_excel(fname, sheet_name=1, engine='openpyxl')
df_Fleet = pd.read_excel(fname, sheet_name=2, engine='openpyxl')



df_nodes.head()
df_requests.head()

n_customers = 25
df_nodes = df_nodes[df_nodes['id'] <= n_customers].copy()

