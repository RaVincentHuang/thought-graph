import pandas as pd
import json
import argparse
import re
import os

class Jsonl2Csv:
    def __init__(self, input_path, save_path,log_path, benchmark_model):
        self.input_path = input_path
        self.save_path = save_path
        self.pattern = r".*-human.jsonl"  
        self.log_root = log_path
        self.benchmark_model = benchmark_model
        self.jsonl_files = self.find_all_jsonl_files()  
        self.merged_file = os.path.join(self.save_path, "merged.jsonl")
        self.label_mapping_file = os.path.join(self.save_path, "label_mapping.csv")
        self.nodes_file = os.path.join(self.save_path, "merged_nodes.jsonl")
        self.edges_file = os.path.join(self.save_path, "merged_edges.jsonl")
        self.nodes_csv = os.path.join(self.save_path, "nodes.csv")
        self.edges_csv = os.path.join(self.save_path, "edges.csv")

    def jsonl_to_csv(self):
        
        os.makedirs(self.save_path, exist_ok=True)
        
        self.merge_files(self.jsonl_files, self.merged_file)
        
        self.gen_nodes_edges(self.merged_file, self.nodes_file, self.edges_file)
        
        self.modify_nodes_file(self.nodes_file)
        
        self.label_mapping(self.nodes_file, self.label_mapping_file)
        
        self.gen_csv()

    
    def modify_nodes_file(self,input_file):
        data_list = []
        with open(input_file,"r",encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                data["state"] = data["state"]
                data["actions"] = data["actions"][1:]
                data["rewards"] = data["rewards"][1:]
                data_list.append(data)
        
        with open(input_file,"w",encoding="utf-8") as file:
            for line in data_list:
                file.write(json.dumps(line, ensure_ascii=False) + "\n")
                

    def find_all_jsonl_files(self):
        target_files = []
        print(self.input_path)
        for root, dirs, files in os.walk(self.input_path):
            folder_name = root.split("/")[-1]
            for file in files:
                if file.endswith(".jsonl"):
                    if re.match(self.pattern, file):
                        cleaned_file = self.clean_jsonl(os.path.join(root, file),data_src=folder_name)
                        success_file = self.success_mark(cleaned_file)
                        target_files.append(success_file)       
        return target_files

    
    def clean_jsonl(self, input_file, data_src):
        raw_data = []
        new_data = []
        folder_name = input_file.split("-human")[0].split("/")[-1]
        log_file = folder_name + "/" + "result.log"
        inits, goals, plans = self.get_init_goal(os.path.join(self.log_root, log_file))
        output_file = input_file.split("-")[0] + "_cleaned.jsonl"
        
        
        with open(input_file, 'r', encoding="utf-8") as f:
            content = f.read()
        
        
        level = 0
        start = 0
        
        for i, char in enumerate(content):
            if char == '{':
                if level == 0:
                    start = i
                level += 1
            elif char == '}':
                level -= 1
                if level == 0:
                    
                    obj_str = content[start:i+1]
                    try:
                        json_obj = json.loads(obj_str)
                        raw_data.append(json_obj)
                    except json.JSONDecodeError as e:
                        pass
        
        
        for data, init, goal, plan in zip(raw_data, inits, goals, plans):
            tmp = data["nodes"][0]["state"]
            tmp.append(init)
            tmp.append(goal)
            tmp.append(plan)
            data["nodes"][0]["state"] = tmp
            for i in range(len(data["nodes"])):
                data["nodes"][i]["state"].append(data_src)
            new_data.append(data)
        
        
        with open(output_file, 'w', encoding="utf-8") as f:
            for obj in new_data:
                f.write(json.dumps(obj) + '\n')
        
        return output_file

    
    def success_mark(self,input_file):
        output_file = input_file.split(".")[0]+ "_success.jsonl"
        folder_name = input_file.split("_cleaned")[0].split("/")[-1]
        log_file = folder_name + "/" + "result.log"
        outputs,plans,corrects = self.get_answer_path(os.path.join(self.log_root,log_file))
        results = self.match_success_path(corrects,plans,input_file)
        with open(output_file,"w",encoding="utf-8") as file:
            for result in results:
                json.dump(result,file,ensure_ascii=False)
                file.write("\n")    
        return output_file

    
    def get_answer_path(self,log_path):
        with open(log_path,"r",encoding="utf-8") as file:
            log_data = file.read()
        output_pattern = re.compile(r"output='([^']*)'")
        plan_pattern = re.compile(r"'plan': '([^']*)'")
        correct_pattern = re.compile(r"correct=(\w+)")

        outputs = output_pattern.findall(log_data)
        plans = plan_pattern.findall(log_data)
        corrects = correct_pattern.findall(log_data)
        
        new_plans = []
        for plan in plans:
            tmp = plan[2:].split("\\n")[:-2]
            new_plans.append(tmp)
        return outputs, new_plans,corrects

    
    def get_init_goal(self,log_path):
        with open(log_path,"r",encoding="utf-8") as file:
            log_data = file.read()
        init_pattern = re.compile(r"'init': '([^']*)'")
        goal_pattern = re.compile(r"'goal': '([^']*)'")
        plan_pattern = re.compile(r"'plan': '([^']*)'")

        inits = init_pattern.findall(log_data)
        goals = goal_pattern.findall(log_data)
        plans = plan_pattern.findall(log_data)

        return inits, goals,plans

    
    def match_success_path(self,corrects, plans, file_path):
        contents = []
        results = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                contents.append(json.loads(line))
        
        for correct, content,plan in zip(corrects, contents,plans):
            
            for node in content["nodes"]:
                node["success"] = False
            
            success_value = True if correct == "True" else False
            if success_value:
                ans_state = plan
                
                
                end_nodes = [node for node in content["nodes"] if node["state"][1] == ans_state]
                
                
                def mark_success_path(node_id):
                    for node in content["nodes"]:
                        if node["node_id"] == node_id:
                            node["success"] = True
                            if node["parent_id"] is not None:
                                mark_success_path(node["parent_id"])
                            break
                
                for end_node in end_nodes:
                    mark_success_path(end_node["node_id"])

            results.append(content)
        
        return results


    
    def merge_files(self, files_path, save_path):
        content = []
        for path in files_path:
            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    content.append(json.loads(line))
        
        with open(save_path, "w", encoding="utf-8") as file:
            for line in content:
                file.write(json.dumps(line, ensure_ascii=False) + "\n")

    def label_mapping(self,input_file,save_path):
        raw_data = []
        target_dict = []
        with open(input_file,"r",encoding="utf-8") as file:
            for line in file:
                raw_data.append(json.loads(line))

        for data in raw_data:
            
            
            tmp_dict = {"node_id":data["node_id"],"node_name":"","cluster_id":"0"}
            action_list = data["state"][1]
            if action_list == [] and data["parent_id"] is None:
                init = data["state"][4]
                goal = data["state"][5]
                plan = data["state"][6]
                
                node_name = init + "%" + goal + "%" + plan
            else:
                node_name = "".join([x+"\n" for x in action_list])
                
            tmp_dict["node_name"] = node_name
            
            target_dict.append(tmp_dict)

        df = pd.DataFrame(target_dict).sort_values(by='node_id')
        df.to_csv(save_path,index=False,encoding="utf-8")
        
    def restore_nodes(self,data,bias=0):
        nodes = data['nodes']
        edges = data['edges']
        max_bias = -1
        
        node_dict = {}
        for node in nodes:
            node_id = node['node_id']
            if node_id not in node_dict:
                node_dict[node_id] = node
        edge_list = []
        
        
        for edge in edges:
            from_node = edge['from'] 
            to_node = edge['to']
            action = edge['action']
            value = edge['value']
            intensity = edge["intensity"] if "intensity" in edge else 1
            max_bias = max(to_node+bias,max_bias)
            new_edge = {
                "Source": from_node+bias,
                "Target": to_node+bias,
                "actions": action,
                "value": value,
                "intensity": intensity
            }
            edge_list.append(new_edge)
            if to_node not in node_dict:
                parent_node = node_dict[from_node]
                parent_state = parent_node['state']
                
                new_state = parent_state[1] + [action]
                new_node = {
                    "node_id": to_node+bias,
                    "parent_id": from_node+bias,
                    "parent_state": parent_state,
                    "state": [parent_state[0] + 1, new_state, False, parent_state[-1]],
                    "actions": [],
                    "rewards": [],
                    "value": value,
                    "success": False
                }
                
                node_dict[to_node] = new_node
            else:
                
                node_dict[to_node]["node_id"] = to_node + bias
                node_dict[to_node]["parent_id"] = from_node + bias

            
            node_dict[0]["node_id"] = bias
        

        
        for node in node_dict.values():
            original_node = next((n for n in nodes if n['node_id'] == node['node_id'] - bias), None)
            if original_node:
                node['success'] = original_node.get('success', False)
        
        return list(node_dict.values()), edge_list, max_bias+1



    def gen_nodes_edges(self,input_file,output_nodes,output_edges):
        restored_nodes = []
        restored_edges = []
        bias = 0
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                nodes, edges, bias = self.restore_nodes(data,bias)
                restored_nodes.extend(nodes)
                restored_edges.extend(edges)

        for node in restored_nodes:
            if node["state"][-1] == self.benchmark_model:
                node["benchmark"] = True
            else:
                node["benchmark"] = False

        with open(output_nodes, 'w', encoding='utf-8') as f:
            for node in restored_nodes:
                f.write(json.dumps(node) + '\n')

        with open(output_edges, 'w', encoding='utf-8') as f:
            for edge in restored_edges:
                f.write(json.dumps(edge) + '\n')

    def gen_nodes_csv(self,input_file,output_file,mapping_dict):
        raw_data = []
        target_data = []
        with open(input_file,"r",encoding="utf-8") as file:
            for line in file:
                raw_data.append(json.loads(line))

        for data in raw_data:
            tmp_dict = {"id":"","node_id":"","step":"","value":"","success":"","benchmark":False,"data_src":"","parent_id":""}
            tmp_dict["id"] = mapping_dict[data["node_id"]]
            tmp_dict["node_id"] = "n{}".format(data["node_id"])
            tmp_dict["step"] = data["state"][0]
            tmp_dict["value"] = data["value"]
            tmp_dict["success"] = data["success"]
            tmp_dict["benchmark"] = data["benchmark"]
            tmp_dict["data_src"] = data["state"][-1]
            if data["parent_id"] is not None:
                tmp_dict["parent_id"] = str("n"+str(data["parent_id"]))
            else:
                tmp_dict["parent_id"] = "root"
            target_data.append(tmp_dict)

        df = pd.DataFrame(target_data)
        df.to_csv(output_file,index=False,encoding="utf-8")

    def gen_edges_csv(self,input_file,output_file,mapping_dict):
        raw_data = []
        target_data = []
        with open(input_file,"r",encoding="utf-8") as file:
            for line in file:
                raw_data.append(json.loads(line))
        
        for data in raw_data:
            tmp_dict = {"Source":"","Target":"","action":"","intensity":"","value":""}
            
            
            tmp_dict["Source"] = "n"+str(data["Source"])
            tmp_dict["Target"] = "n"+str(data["Target"])
            tmp_dict["action"] = data["actions"]
            tmp_dict["intensity"] = data["intensity"]
            tmp_dict["value"] = data["value"]
            target_data.append(tmp_dict)

        df = pd.DataFrame(target_data)
        df.to_csv(output_file,index=False,encoding="utf-8")

    
    def get_mapping_dict(self,input_file):
        mapping_dict = {}
        df = pd.read_csv(input_file)
        for index,row in df.iterrows():
            mapping_dict[row["node_id"]] = row["node_name"]
        return mapping_dict
    
    
    def gen_csv(self):
        mapping_dict = self.get_mapping_dict(self.label_mapping_file)
        self.gen_nodes_csv(self.nodes_file,self.nodes_csv,mapping_dict)
        self.gen_edges_csv(self.edges_file,self.edges_csv,mapping_dict)

def main(args):
    instance = Jsonl2Csv(args.input_path, args.save_path,args.log_path,args.benchmark_model)
    instance.jsonl_to_csv()

def data_process(input_path,save_path,log_path,benchmark_model):
    instance = Jsonl2Csv(input_path, save_path,log_path,benchmark_model)
    instance.jsonl_to_csv()
    return instance.nodes_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL to CSV")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--save_path", type=str, required=True, help="Path to the output CSV file")
    parser.add_argument("--log_path", type=str, required=True, help="Path to the log file")
    parser.add_argument("--benchmark_model", type=str, required=True, help="Path to the benchmark file")
    args = parser.parse_args()
    main(args)