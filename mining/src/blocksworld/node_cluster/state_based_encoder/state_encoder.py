from .stateEncoder import state_encoder
import pandas as pd
from tqdm import tqdm

def encoder(file_path: str, save_path: str):
    df = pd.read_csv(file_path)
    groups = [] 
    current_group = [] 
    for index, row in df.iterrows():
        if "the hand is empty" in row["id"]:
            if current_group:
                groups.append(current_group)
            current_group = []
        current_group.append(row)
    if current_group:
        groups.append(current_group)
    
    for group in tqdm(groups, desc="encoding"):
        
        root_node = group[0]
        tmp = root_node["id"]
        init_state, goal_state, plans = tmp.split("%")[0].strip(), tmp.split("%")[1].strip(), tmp.split("%")[2].strip()
        plans = plans.split("\\n")[1:-2]
        root_node["similarity"] = state_encoder(init_state, goal_state, plans, [""])
        if root_node["similarity"] == -1 or root_node["similarity"] == -2:
            pass
        else:
            if len(root_node["similarity"]) < 6:
                root_node["similarity"] = "0" * (6 - len(root_node["similarity"])) + root_node["similarity"]

        df.loc[df["node_id"] == root_node["node_id"], "similarity"] = str(root_node["similarity"])
        
        for node in group[1:]:
            actions_list = node["id"].strip().split("\n")
            node["similarity"] = state_encoder(init_state, goal_state, plans, actions_list)
            if node["similarity"] == -1 or node["similarity"] == -2:
                pass
            else:
                if len(node["similarity"]) < 6:
                    node["similarity"] = "0" * (6 - len(node["similarity"])) + node["similarity"]
            df.loc[df["node_id"] == node["node_id"], "similarity"] = str(node["similarity"])

    df.to_csv(save_path, index=False)
    print("Encoding completed, file saved")
