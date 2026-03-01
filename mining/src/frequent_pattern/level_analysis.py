import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher

def read_pattern_graph(filename):
    patterns = []
    current_pattern = None
    
    with open(filename, 'r') as f:
        for line in f:
            if line.strip().endswith(':'):
                if current_pattern is not None:
                    patterns.append(current_pattern)
                current_pattern = nx.DiGraph()  
            elif line.startswith('v'):
                _, id, label = line.split()
                id, label = int(id), int(label)
                current_pattern.add_node(id, label=label)
            elif line.startswith('e'):
                _, u, v, w = line.split()
                u, v, w = int(u), int(v), float(w)
                current_pattern.add_edge(u, v, label=w)  
    
    if current_pattern is not None:
        patterns.append(current_pattern)
    
    return patterns

def read_graph(filename):
    G = nx.DiGraph()  
    levels = {}
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('v'):
                _, id, label, level, success = line.split()
                id, label, level = int(id), int(label), int(level)
                G.add_node(id, label=label)
                levels[id] = level
            elif line.startswith('e'):
                _, u, v, w = line.split()
                u, v, w = int(u), int(v), float(w)
                G.add_edge(u, v, label=w)  
    
    return G, levels

def calculate_pattern_level(data_graph, data_levels, pattern_graph):
    matcher = DiGraphMatcher(data_graph, pattern_graph,
                           node_match=lambda n1, n2: n1['label'] == n2['label'])
    
    root_in_pattern = None
    for node in pattern_graph.nodes():
        if pattern_graph.in_degree(node) == 0:
            root_in_pattern = node
            break
    
    if root_in_pattern is None:
        print("Error: No root node found in pattern graph!")
        return 0
    
    match_count = 0
    total_level = 0
    
    for mapping in matcher.subgraph_isomorphisms_iter():
        reverse_mapping = {v: k for k, v in mapping.items()}
        
        valid_match = True
        for u, v in pattern_graph.edges():
            mapped_u = reverse_mapping[u]
            mapped_v = reverse_mapping[v]
            if not data_graph.has_edge(mapped_u, mapped_v):
                valid_match = False
                break
        
        if not valid_match:
            continue
            
        match_count += 1
        root_node = reverse_mapping[root_in_pattern]
        root_level = data_levels[root_node]
        total_level += root_level

    if match_count == 0:
        return 0
    
    avg_level = total_level / match_count
    return avg_level

def evaluate_patterns(data_filename, pattern_filename):
    data_graph, data_levels = read_graph(data_filename)
    
    pattern_graphs = read_pattern_graph(pattern_filename)
    
    results = {}
    for i, pattern_graph in enumerate(pattern_graphs):
        avg_level = calculate_pattern_level(data_graph, data_levels, pattern_graph)
        results[f'Pattern {i}'] = avg_level
    
    return results

def analyze_pattern_metrics(data_file, pattern_file):
    results = evaluate_patterns(data_file, pattern_file)
    if not results:
        print("Warning: No results")
        return 0.0
    
    total_level = sum(level for level in results.values())
    result_count = len(results)
    avg_level = round(total_level / result_count, 2)
    
    return avg_level

