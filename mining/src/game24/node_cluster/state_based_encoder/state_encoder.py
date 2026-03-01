import pandas as pd
from tqdm import tqdm
import os

def encoder(file_path: str, save_path: str):
    
    if os.path.exists(save_path):
        return

    df = pd.read_csv(file_path)
        
    groups = [] 
    current_group = [] 
    for index, row in df.iterrows():
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
            current_state = node["current_state"]
            if 1:
                def leverage_encode(results_list, target):
                    
                    result_str = ""
                    
                    for result in results_list:
                        if abs(result - target) <= 1e-3:
                            result_str += "11"
                        elif abs(result - target) <= 5:
                            result_str += "10"
                        elif abs(result - target) <= 10:
                            result_str += "01"
                        else:
                            result_str += "00"
                    return result_str

                def encode_final(nums_list, target):
                    length = len(nums_list)
                    if length == 0:
                        raise ValueError("nums_list is empty")
                    elif length == 1:
                        if nums_list[0] == target:
                            return "11111111"
                            
                        else:
                            return "00000000"
                            
                    elif length == 2:
                        bigger = max(nums_list)
                        smaller = min(nums_list)
                        plus_result = bigger + smaller
                        minus_result = bigger - smaller
                        multi_result = bigger * smaller
                        div_result = round(float(bigger) / float(smaller)) if smaller != 0 else -1
                        return leverage_encode([plus_result, minus_result, multi_result, div_result], target)
                        
                    elif length == 3:
                        
                        result_list = []
                        for i in range(length):
                            for j in range(i + 1, length):
                                
                                a, b = nums_list[i], nums_list[j]
                                remaining = nums_list[3 - i - j]
                                plus_result = a + b
                                minus_result = a - b
                                multi_result = a * b
                                div_result = round(float(a) / float(b)) if b != 0 else -1
                                tmp_list = [
                                    (plus_result + remaining, abs(plus_result - remaining), plus_result * remaining, abs(round(float(plus_result) / float(remaining)) if remaining != 0 else -1)),
                                    (minus_result + remaining, abs(minus_result - remaining), minus_result * remaining, abs(round(float(minus_result) / float(remaining)) if remaining != 0 else -1)),
                                    (multi_result + remaining, abs(multi_result - remaining), multi_result * remaining, abs(round(float(multi_result) / float(remaining)) if remaining != 0 else -1))
                                ]
                                
                                if div_result != -1:
                                    tmp_list.append((
                                        div_result + remaining,
                                        abs(div_result - remaining),
                                        div_result * remaining,
                                        abs(round(float(div_result) / float(remaining)) if remaining != 0 else -1)
                                    ))
                                else:
                                    tmp_list.append((-1, -1, -1, -1))
                                
                                result_list.extend(tmp_list)        
                        final_result = [0,0,0,0]
                        for item in result_list:
                            for i in range(4):
                                if abs(item[i] - target) < abs(final_result[i] - target):
                                    final_result[i] = item[i]
                        return leverage_encode(final_result, target)
                    else:
                        return "00000000"


                if node["action"] == "root":
                    node["similarity"] = "-1"
                elif node["action"] == "BAD_FORMAT!!!":
                    node["similarity"] = "-2"
                else:
                    current_state = sorted([round(float(x)) for x in current_state.split(" ")])
                    node["similarity"] = encode_final(current_state, 24)

                
            df.loc[df["node_id"] == node["node_id"], "similarity"] = str(node["similarity"])
    df.to_csv(save_path, index=False)
