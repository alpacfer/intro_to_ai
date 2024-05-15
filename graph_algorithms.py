# graph_algorithms.py
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import collections
from collections import deque
import pandas as pd

# Function to visualize the graph
def visualize_graph(graph, heuristics, start_node, goal_nodes, visited_nodes=None, path_nodes=None):
    G = nx.Graph()
    for node, edges in graph.items():
        for neighbor, cost in edges.items():
            G.add_edge(node, neighbor, weight=cost)

    pos = nx.spring_layout(G, seed=42)  # Using a seed for reproducibility
    pos[start_node][0] = -1.0  # Place start node on the far left
    for goal_node in goal_nodes:
        pos[goal_node][0] = 1.0  # Place goal nodes on the far right

    min_x = min(pos[node][0] for node in pos)
    max_x = max(pos[node][0] for node in pos)
    for node in pos:
        pos[node][0] = (pos[node][0] - min_x) / (max_x - min_x) * 2 - 1  # Scale to [-1, 1]

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    labels = {node: f"{node}\n$h={heuristics[node]}$" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')
    
    if visited_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=visited_nodes, node_color='lightyellow', node_size=3000)
        
    if path_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=path_nodes, node_color='lightgreen', node_size=3000)

    nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_color='green', node_size=3000)
    nx.draw_networkx_nodes(G, pos, nodelist=goal_nodes, node_color='red', node_size=3000)

    plt.title("Graph Visualization with Start, Goal, Visited, and Path Nodes")
    plt.show()

# A* Algorithm
def astar(graph, heuristics, start, goal_nodes):
    # Priority queue: (cost, current_node, path)
    frontier = [(heuristics[start], 0, start, [start])]  # (f, g, node, path)
    explored = set()
    visited_nodes = set()
    iterations = 0
    
    # Data collection for pandas DataFrame
    data = {
        "Iteration": [],
        "Node": [],
        "Queue": []
    }

    while frontier:
        iterations += 1
        # Pop the node with the smallest f value
        _, current_cost, current_node, path = heapq.heappop(frontier)

        # Goal test when the node is selected for expansion
        if current_node in goal_nodes:
            data["Iteration"].append(iterations)
            data["Node"].append(current_node)
            data["Queue"].append([(f, node) for f, _, node, _ in frontier])
            df = pd.DataFrame(data)
            print(f"Goal {current_node} reached! Path: {path}, Cost: {current_cost}")
            return path, visited_nodes, df
        
        if current_node not in explored:
            explored.add(current_node)
            visited_nodes.add(current_node)

            # Expand the node and add to frontier
            for neighbor, cost in graph[current_node].items():
                if neighbor not in explored:
                    new_cost = current_cost + cost
                    f = new_cost + heuristics[neighbor]
                    new_path = list(path)
                    new_path.append(neighbor)
                    heapq.heappush(frontier, (f, new_cost, neighbor, new_path))
        
        # Collect data for the current iteration
        data["Iteration"].append(iterations)
        data["Node"].append(current_node)
        data["Queue"].append([(f, node) for f, _, node, _ in frontier])
    
    print("Goal not reached.")
    df = pd.DataFrame(data)
    return None, visited_nodes, df

# Greedy Best-First Search (GBFS) Algorithm
def gbfs(graph, heuristics, start, goal_nodes):
    # Priority queue: (heuristic, current_node, path)
    frontier = [(heuristics[start], start, [start])]
    explored = set()
    visited_nodes = set()
    iterations = 0
    
    # Data collection for pandas DataFrame
    data = {
        "Iteration": [],
        "Node": [],
        "Queue": []
    }

    while frontier:
        iterations += 1
        # Pop the node with the smallest heuristic value
        _, current_node, path = heapq.heappop(frontier)

        # Goal test when the node is selected for expansion
        if current_node in goal_nodes:
            data["Iteration"].append(iterations)
            data["Node"].append(current_node)
            data["Queue"].append([(heuristics[node], node) for _, node, _ in frontier])
            df = pd.DataFrame(data)
            print(f"Goal {current_node} reached! Path: {path}")
            return path, visited_nodes, df
        
        if current_node not in explored:
            explored.add(current_node)
            visited_nodes.add(current_node)

            # Expand the node and add to frontier based on heuristic value
            for neighbor in graph[current_node]:
                if neighbor not in explored:
                    new_path = list(path)
                    new_path.append(neighbor)
                    heapq.heappush(frontier, (heuristics[neighbor], neighbor, new_path))
        
        # Collect data for the current iteration
        data["Iteration"].append(iterations)
        data["Node"].append(current_node)
        data["Queue"].append([(heuristics[node], node) for _, node, _ in frontier])
    
    print("Goal not reached.")
    df = pd.DataFrame(data)
    return None, visited_nodes, df

# Function to compute shortest path costs
def compute_shortest_path_costs(graph, goal_nodes):
    G = nx.Graph()
    for node, edges in graph.items():
        for neighbor, cost in edges.items():
            G.add_edge(node, neighbor, weight=cost)
    
    shortest_path_costs = {}
    for goal in goal_nodes:
        lengths = nx.single_source_dijkstra_path_length(G, goal)
        for node, cost in lengths.items():
            if node not in shortest_path_costs or cost < shortest_path_costs[node]:
                shortest_path_costs[node] = cost
    
    return shortest_path_costs

# Function to check admissibility
def check_admissibility(heuristics, shortest_path_costs):
    admissible = True
    for node in heuristics:
        h_value = heuristics[node]
        shortest_path_cost = shortest_path_costs.get(node, float('inf'))
        if h_value > shortest_path_cost:
            admissible = False
            print(f"Heuristic is not admissible at node {node}: h(n) = {h_value} > actual cost = {shortest_path_cost}")
    return admissible

def display_minimax_tree(game_tree, values, best_path, root='A'):
    G = nx.DiGraph()

    def add_edges(graph, node):
        if isinstance(game_tree[node], int):
            graph.add_node(node, label=f"{node}\n{values[node]}")
        else:
            graph.add_node(node, label=f"{node}\n{values[node]}")
            for child in game_tree[node]:
                graph.add_edge(node, child)
                add_edges(graph, child)

    add_edges(G, root)
    
    def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
        pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
        return pos

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=[]):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        
        children = list(G.neighbors(root))
        if not isinstance(game_tree[root], int):
            width = width / len(children)
            nextx = xcenter - width / 2 - width / (2 * len(children))
            for child in children:
                nextx += width / len(children)
                pos = _hierarchy_pos(G, child, width, vert_gap, vert_loc - vert_gap, nextx, pos, root, parsed)
        return pos

    pos = hierarchy_pos(G, root)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', arrowsize=20)
    nx.draw_networkx_nodes(G, pos, nodelist=best_path, node_color='lightgreen', node_size=3000)

    def display_minimax_values(node, game_tree, is_maximizing):
        if not isinstance(game_tree[node], int):
            x, y = pos[node]
            plt.text(x, y + 0.05, s=f"{values[node]}", bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'), horizontalalignment='center')
            for child in game_tree[node]:
                display_minimax_values(child, game_tree, not is_maximizing)

    display_minimax_values(root, game_tree, True)
    plt.title("Game Tree with Minimax Values and Best Path")
    plt.show()


def minimax(node, depth, maximizing_player, game_tree, values):
    # If the node is a terminal node (no children), return its value and the path
    if not isinstance(game_tree[node], list):
        values[node] = game_tree[node]
        return game_tree[node], [node]

    if maximizing_player:
        max_eval = float('-inf')
        best_path = []
        for child in game_tree[node]:
            eval, path = minimax(child, depth + 1, False, game_tree, values)
            if eval > max_eval:
                max_eval = eval
                best_path = [node] + path
        values[node] = max_eval
        return max_eval, best_path
    else:
        min_eval = float('inf')
        best_path = []
        for child in game_tree[node]:
            eval, path = minimax(child, depth + 1, True, game_tree, values)
            if eval < min_eval:
                min_eval = eval
                best_path = [node] + path
        values[node] = min_eval
        return min_eval, best_path

def print_ordered_values(values):
    ordered_nodes = sorted(values.items())
    for node, value in ordered_nodes:
        print(f"Node {node}: {value}")

# Alpha beta pruning algorithm
def alpha_beta_pruning(node, depth, alpha, beta, maximizing_player, game_tree, values, alpha_values, beta_values, pruned_branches):
    # If the node is a terminal node (no children), return its value and the path
    if not isinstance(game_tree[node], list):
        values[node] = game_tree[node]
        alpha_values[node] = alpha
        beta_values[node] = beta
        return game_tree[node], [node]

    if maximizing_player:
        max_eval = float('-inf')
        best_path = []
        for child in game_tree[node]:
            eval, path = alpha_beta_pruning(child, depth + 1, alpha, beta, False, game_tree, values, alpha_values, beta_values, pruned_branches)
            if eval > max_eval:
                max_eval = eval
                best_path = [node] + path
            alpha = max(alpha, eval)
            if beta <= alpha:
                # Collect all nodes in the pruned subtree
                def collect_pruned_nodes(n):
                    if isinstance(game_tree[n], list):
                        for ch in game_tree[n]:
                            collect_pruned_nodes(ch)
                    pruned_branches.append(n)
                collect_pruned_nodes(child)
                break
        values[node] = max_eval
        alpha_values[node] = alpha
        beta_values[node] = beta
        return max_eval, best_path
    else:
        min_eval = float('inf')
        best_path = []
        for child in game_tree[node]:
            eval, path = alpha_beta_pruning(child, depth + 1, alpha, beta, True, game_tree, values, alpha_values, beta_values, pruned_branches)
            if eval < min_eval:
                min_eval = eval
                best_path = [node] + path
            beta = min(beta, eval)
            if beta <= alpha:
                # Collect all nodes in the pruned subtree
                def collect_pruned_nodes(n):
                    if isinstance(game_tree[n], list):
                        for ch in game_tree[n]:
                            collect_pruned_nodes(ch)
                    pruned_branches.append(n)
                collect_pruned_nodes(child)
                break
        values[node] = min_eval
        alpha_values[node] = alpha
        beta_values[node] = beta
        return min_eval, best_path


# Function to display the alpha-beta tree
def display_alpha_beta_tree(game_tree, values, alpha_values, beta_values, best_path, root='A'):
    G = nx.DiGraph()

    def add_edges(graph, node):
        if isinstance(game_tree[node], int):
            values[node] = values.get(node, game_tree[node])
            alpha_values[node] = alpha_values.get(node, 'N/A')
            beta_values[node] = beta_values.get(node, 'N/A')
            graph.add_node(node, label=f"{node}\nα={alpha_values[node]}\nβ={beta_values[node]}\nv={values[node]}")
        else:
            values[node] = values.get(node, 'N/A')
            alpha_values[node] = alpha_values.get(node, 'N/A')
            beta_values[node] = beta_values.get(node, 'N/A')
            graph.add_node(node, label=f"{node}\nα={alpha_values[node]}\nβ={beta_values[node]}\nv={values[node]}")
            for child in game_tree[node]:
                graph.add_edge(node, child)
                add_edges(graph, child)

    add_edges(G, root)
    
    def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
        pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
        return pos

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=[]):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        
        children = list(G.neighbors(root))
        if not isinstance(game_tree[root], int):
            width = width / len(children)
            nextx = xcenter - width / 2 - width / (2 * len(children))
            for child in children:
                nextx += width / len(children)
                pos = _hierarchy_pos(G, child, width, vert_gap, vert_loc - vert_gap, nextx, pos, root, parsed)
        return pos

    pos = hierarchy_pos(G, root)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', arrowsize=20)
    nx.draw_networkx_nodes(G, pos, nodelist=best_path, node_color='lightgreen', node_size=3000)

    pruned_nodes = [node for node in G.nodes if alpha_values[node] == 'N/A' or beta_values[node] == 'N/A']
    nx.draw_networkx_nodes(G, pos, nodelist=pruned_nodes, node_color='red', node_size=3000, alpha=0.5)

    plt.title("Game Tree with Alpha-Beta Pruning Values and Best Path")
    plt.show()

# BFS Algorithm
def bfs(graph, start, goal):
    frontier = deque([(start, [start])])  # Using deque for the queue
    explored = set()
    order_of_exploration = []
    iterations = 0

    # Data collection for pandas DataFrame
    data = {
        "Iteration": [],
        "Node": [],
        "Queue": []
    }

    while frontier:
        iterations += 1
        current_node, path = frontier.popleft()  # Pop from the left (FIFO)

        if current_node == goal:
            order_of_exploration.append(current_node)
            data["Iteration"].append(iterations)
            data["Node"].append(current_node)
            data["Queue"].append([node for node, _ in frontier])
            df = pd.DataFrame(data)
            print(f"Goal {goal} reached! Path: {path}")
            return path, order_of_exploration, df
        
        if current_node not in explored:
            explored.add(current_node)
            order_of_exploration.append(current_node)

            # Expand the node and add to frontier in FIFO manner
            for neighbor in graph[current_node]:
                if neighbor not in explored:
                    new_path = list(path)
                    new_path.append(neighbor)
                    frontier.append((neighbor, new_path))  # Append to the right (FIFO)

            # Collect data for the current iteration
            data["Iteration"].append(iterations)
            data["Node"].append(current_node)
            data["Queue"].append([node for node, _ in frontier])

    print("Goal not reached.")
    df = pd.DataFrame(data)
    return None, order_of_exploration, df


def dfs(graph, start, goal):
    stack = [(start, [start])]  # Using list for the stack
    visited = set()
    order_of_exploration = []
    iterations = 0

    # Data collection for pandas DataFrame
    data = {
        "Iteration": [],
        "Node": [],
        "Stack": []
    }

    while stack:
        iterations += 1
        current_node, path = stack.pop()  # Pop from the right (LIFO)

        if current_node == goal:
            order_of_exploration.append(current_node)
            data["Iteration"].append(iterations)
            data["Node"].append(current_node)
            data["Stack"].append([node for node, _ in stack])
            df = pd.DataFrame(data)
            print(f"Goal {goal} reached! Path: {path}")
            return path, order_of_exploration, df

        if current_node not in visited:
            visited.add(current_node)
            order_of_exploration.append(current_node)

            # Expand the node and add to stack in LIFO manner
            for neighbor in graph[current_node]:  # Maintain the original order
                if neighbor not in visited:
                    new_path = list(path)
                    new_path.append(neighbor)
                    stack.append((neighbor, new_path))  # Append to the right (LIFO)

            # Collect data for the current iteration
            data["Iteration"].append(iterations)
            data["Node"].append(current_node)
            data["Stack"].append([node for node, _ in stack])

    print("Goal not reached.")
    df = pd.DataFrame(data)
    return None, order_of_exploration, df


# Visualization Function for BFS and DFS
def visualize_search(graph, start_node, goal_node, path_nodes, explored_nodes):
    G = nx.DiGraph()
    for node, edges in graph.items():
        for neighbor in edges:
            G.add_edge(node, neighbor)

    def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
        pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
        return pos

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=[]):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        
        children = list(G.neighbors(root))
        if len(children) != 0:
            width = width / len(children)
            nextx = xcenter - width / 2 - width / (2 * len(children))
            for child in children:
                nextx += width / len(children)
                pos = _hierarchy_pos(G, child, width, vert_gap, vert_loc - vert_gap, nextx, pos, root, parsed)
        return pos

    pos = hierarchy_pos(G, start_node)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
    
    # Draw the nodes with specific colors
    nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_color='green', node_size=3000)
    nx.draw_networkx_nodes(G, pos, nodelist=[goal_node], node_color='darkgreen', node_size=3000)  # Strong green for goal node
    nx.draw_networkx_nodes(G, pos, nodelist=[node for node in explored_nodes if node != goal_node], node_color='lightgreen', node_size=3000)
    nx.draw_networkx_nodes(G, pos, nodelist=[node for node in path_nodes if node != goal_node], node_color='#FFD700', node_size=3000)  # Stronger yellow for path nodes

    # Adding numbers to indicate the order in which nodes were visited
    node_labels = {node: str(idx + 1) for idx, node in enumerate(explored_nodes)}
    for node, (x, y) in pos.items():
        if node in node_labels:
            plt.text(x, y - 0.05, s=node_labels[node], bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'), 
                     horizontalalignment='center', fontsize=8)

    plt.title("Graph Visualization with BFS/DFS Search Path and Explored Nodes")
    plt.show()


def ucs(graph, start, goal_nodes):
    # Priority queue: (cost, current_node, path)
    frontier = [(0, start, [start])]
    explored = set()
    visited_nodes = set()
    iterations = 0
    
    # Data collection for pandas DataFrame
    data = {
        "Iteration": [],
        "Node": [],
        "Queue": []
    }

    while frontier:
        iterations += 1
        # Pop the node with the smallest cost
        current_cost, current_node, path = heapq.heappop(frontier)

        # Goal test when the node is selected for expansion
        if current_node in goal_nodes:
            data["Iteration"].append(iterations)
            data["Node"].append(current_node)
            data["Queue"].append([(cost, node) for cost, node, _ in frontier])
            df = pd.DataFrame(data)
            print(f"Goal {current_node} reached! Path: {path}, Cost: {current_cost}")
            return path, visited_nodes, df
        
        if current_node not in explored:
            explored.add(current_node)
            visited_nodes.add(current_node)

            # Expand the node and add to frontier
            for neighbor, cost in graph[current_node].items():
                if neighbor not in explored:
                    new_cost = current_cost + cost
                    new_path = list(path)
                    new_path.append(neighbor)
                    heapq.heappush(frontier, (new_cost, neighbor, new_path))
        
        # Collect data for the current iteration
        data["Iteration"].append(iterations)
        data["Node"].append(current_node)
        data["Queue"].append([(cost, node) for cost, node, _ in frontier])
    
    print("Goal not reached.")
    df = pd.DataFrame(data)
    return None, visited_nodes, df
