# graph_algorithms.py
import networkx as nx
import matplotlib.pyplot as plt
import heapq

# Function to visualize the graph
def visualize_graph(graph, heuristics, start_node, goal_nodes, path_nodes=None):
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

    nx.draw_networkx_nodes(G, pos, nodelist=[start_node], node_color='green', node_size=3000)
    nx.draw_networkx_nodes(G, pos, nodelist=goal_nodes, node_color='red', node_size=3000)
    
    if path_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=path_nodes, node_color='lightgreen', node_size=3000)

    plt.title("Graph Visualization with Start and Goal Nodes and Heuristics")
    plt.show()

# A* Algorithm
def a_star(graph, heuristics, start, goals):
    open_list = []
    heapq.heappush(open_list, (heuristics[start], 0, start, [start]))
    closed_list = set()
    visited_nodes = []

    while open_list:
        f_value, g_value, current_node, path = heapq.heappop(open_list)
        if current_node in closed_list:
            continue
        
        visited_nodes.append((current_node, g_value, heuristics[current_node], f_value))
        print(f"Visiting node: {current_node}, f(n): {f_value}, g(n): {g_value}, h(n): {heuristics[current_node]}, Path: {path}")
        closed_list.add(current_node)
        
        if current_node in goals:
            print(f"\nGoal {current_node} reached! Total cost: {g_value}, Path: {path}")
            print(f"Visited nodes: {visited_nodes}\n")
            return path
        
        for neighbor, step_cost in graph[current_node].items():
            if neighbor not in closed_list:
                new_g_value = g_value + step_cost
                f_value = new_g_value + heuristics[neighbor]
                heapq.heappush(open_list, (f_value, new_g_value, neighbor, path + [neighbor]))

    print("No path found to the goal.")
    return None

# Greedy Best-First Search (GBFS) Algorithm
def gbfs(graph, heuristics, start, goals):
    open_list = []
    heapq.heappush(open_list, (heuristics[start], start, [start]))
    closed_list = set()
    visited_nodes = []

    while open_list:
        h_value, current_node, path = heapq.heappop(open_list)
        if current_node in closed_list:
            continue
        
        visited_nodes.append(current_node)
        print(f"Visiting node: {current_node}, h(n): {h_value}, Path: {path}")
        closed_list.add(current_node)
        
        if current_node in goals:
            print(f"\nGoal {current_node} reached! Path: {path}")
            print(f"Visited nodes: {visited_nodes}\n")
            return path
        
        for neighbor in graph[current_node]:
            if neighbor not in closed_list:
                heapq.heappush(open_list, (heuristics[neighbor], neighbor, path + [neighbor]))

    print("No path found to the goal.")
    return None

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

def display_minimax_tree(game_tree, best_path, root='A'):
    G = nx.DiGraph()

    def add_edges(graph, node):
        if isinstance(game_tree[node], int):
            graph.add_node(node, label=f"{node}\n{game_tree[node]}")
        else:
            graph.add_node(node, label=node)
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
            value, _ = minimax(node, game_tree, is_maximizing)
            x, y = pos[node]
            plt.text(x, y + 0.1, s=f"{value}", bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'), horizontalalignment='center')
            for child in game_tree[node]:
                display_minimax_values(child, game_tree, not is_maximizing)

    display_minimax_values(root, game_tree, True)
    plt.title("Game Tree with Minimax Values and Best Path")
    plt.show()


def minimax(node, game_tree, is_maximizing):
    if isinstance(game_tree[node], int):
        return game_tree[node], [node]

    if is_maximizing:
        best_value = float('-inf')
        best_path = []
        for child in game_tree[node]:
            val, path = minimax(child, game_tree, False)
            if val > best_value:
                best_value = val
                best_path = path
        return best_value, [node] + best_path
    else:
        best_value = float('inf')
        best_path = []
        for child in game_tree[node]:
            val, path = minimax(child, game_tree, True)
            if val < best_value:
                best_value = val
                best_path = path
        return best_value, [node] + best_path


# Alpha-beta pruning algorithm
def alpha_beta(node, game_tree, is_maximizing, alpha, beta, alpha_beta_data):
    if isinstance(game_tree[node], int):
        alpha_beta_data['visited_nodes'].add(node)
        return game_tree[node], [node]

    if is_maximizing:
        best_value = float('-inf')
        best_path = []
        for child in game_tree[node]:
            val, path = alpha_beta(child, game_tree, False, alpha, beta, alpha_beta_data)
            if val > best_value:
                best_value = val
                best_path = path
            alpha = max(alpha, best_value)
            if beta <= alpha:
                break  # Beta cut-off
        alpha_beta_data['alpha'][node] = alpha
        alpha_beta_data['beta'][node] = beta
        alpha_beta_data['visited_nodes'].add(node)
        return best_value, [node] + best_path
    else:
        best_value = float('inf')
        best_path = []
        for child in game_tree[node]:
            val, path = alpha_beta(child, game_tree, True, alpha, beta, alpha_beta_data)
            if val < best_value:
                best_value = val
                best_path = path
            beta = min(beta, best_value)
            if beta <= alpha:
                break  # Alpha cut-off
        alpha_beta_data['alpha'][node] = alpha
        alpha_beta_data['beta'][node] = beta
        alpha_beta_data['visited_nodes'].add(node)
        return best_value, [node] + best_path

# Function to display the alpha-beta tree
def display_alpha_beta_tree(game_tree, best_path, alpha_beta_data, root='A'):
    G = nx.DiGraph()

    def add_edges(graph, node):
        if isinstance(game_tree[node], int):
            graph.add_node(node, label=f"{node}\n{game_tree[node]}")
        else:
            graph.add_node(node, label=node)
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
    nx.draw_networkx_nodes(G, pos, nodelist=best_path, node_color='green', node_size=3000)
    explored_nodes = list(alpha_beta_data['visited_nodes'] - set(best_path))
    nx.draw_networkx_nodes(G, pos, nodelist=explored_nodes, node_color='lightgreen', node_size=3000)

    for node in G.nodes:
        if node in alpha_beta_data['visited_nodes']:
            alpha = alpha_beta_data['alpha'].get(node, None)
            beta = alpha_beta_data['beta'].get(node, None)
            x, y = pos[node]
            if alpha is not None and beta is not None:
                plt.text(x, y + 0.1, s=f"α={alpha}, β={beta}", bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'), horizontalalignment='center')

    plt.title("Game Tree with Alpha-Beta Pruning Values, Alpha and Beta Values, and Exploration Marking")
    plt.show()
