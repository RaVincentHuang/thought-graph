import networkx as nx

def read_graph(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    graphs = []
    current_graph = None

    for line in lines:
        line = line.strip()
        if line.endswith(':'):
            if current_graph:
                graphs.append(current_graph)
            current_graph = nx.Graph()
        elif line.startswith('v'):
            _, vertex_id, vertex_lable = line.split()
            current_graph.add_node(int(vertex_id), coord=int(vertex_lable))
        elif line.startswith('e'):
            _, start_vertex, end_vertex, edge_lable = line.split()
            current_graph.add_edge(int(start_vertex), int(end_vertex), label=int(edge_lable))  # Ensure consistency with 'label'

    if current_graph:
        graphs.append(current_graph)

    return graphs

def analyze_edge_and_path_metrics(filename):
    try:
        graphs = read_graph(filename)
        if not graphs:
            print(f"Warning: No valid graphs found in file {filename}")
            return 0, 0
            
        edge_counts = [len(g.edges()) for g in graphs]
        avg_edges = sum(edge_counts) / len(edge_counts) if edge_counts else 0
        
        max_path_lengths = []
        for g in graphs:
            try:
                path_length = nx.diameter(nx.Graph(g.edges()))
                max_path_lengths.append(path_length)
            except (nx.NetworkXError, nx.NetworkXNoPath):
                continue
                
        max_path = max(max_path_lengths) if max_path_lengths else 0
        
        return round(avg_edges, 2), max_path
        
    except Exception as e:
        print(f"Error analyzing file {filename}: {str(e)}")
        return 0, 0