import pandas as pd
from tqdm import tqdm

def encoder(file_path: str, save_path: str):
    df = pd.read_csv(file_path)
    groups = [] 
    current_group = [] 
    for _, row in df.iterrows():
        if row["parent_id"] == "root":
            if current_group:
                groups.append(current_group)
            current_group = []
        current_group.append(row)
    if current_group:
        groups.append(current_group)
    
    for group in tqdm(groups):
        root_node = group[0]
        df.loc[df["node_id"] == root_node["node_id"], "similarity"] = "root"
        for node in group[1:]:
            acc,logi,dis,infer = node["acc"],node["logi"],node["dis"],node["infer"]
            node["similarity"] = str(dis) + "_" + str(infer)
            df.loc[df["node_id"] == node["node_id"], "similarity"] = str(node["similarity"])
    
    
    df.to_csv(save_path, index=False)
