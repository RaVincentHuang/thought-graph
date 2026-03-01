from config import results_dir, lg_output_path, lg_data_dir
import subprocess
import os
import shutil
import json
from datetime import datetime
import pandas as pd
from edge_count_analysis import analyze_edge_and_path_metrics
from level_analysis import analyze_pattern_metrics
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import argparse

class ExperimentManager:
    def __init__(self, exp_name, support_values):
        """Initialize experiment manager
        
        Args:
            exp_name: Experiment name
            support_values: List of support values
        """
        self.exp_name = exp_name
        self.support_values = support_values
        self.exp_description = self._generate_description()
        self.completed_supports = []
        
    def _generate_description(self):
        """Generate experiment description"""
        return "Support values: " + ", ".join(map(str, self.support_values))
        
    def initialize(self):
        global raw_data_dir
        """Initialize experiment environment"""
        self.completed_supports = self.exp_init(self.exp_name, self.exp_description)
        path_prefix = os.getcwd()
        print(raw_data_dir)
        raw_data_dir = path_prefix + "/" + raw_data_dir
        for file in os.listdir(raw_data_dir):
            if file.endswith(".txt"):
                base_name = os.path.splitext(file)[0]
                lg_file = os.path.join(raw_data_dir, f"{base_name}.lg")
                if not os.path.exists(lg_file):
                    self.trans_to_grami(os.path.join(raw_data_dir, file))
                else:
                    print(f"Skipping existing lg file: {lg_file}")
        for file in os.listdir(lg_data_dir):
            os.remove(os.path.join(lg_data_dir, file))
        for file in os.listdir(raw_data_dir):
            if file.endswith(".lg"):
                shutil.copy(os.path.join(raw_data_dir, file), os.path.join(lg_data_dir, file))


    @staticmethod
    def trans_to_grami(file_path):
        file_name = file_path.split(".")[0]
        new_lines = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("e"):
                    parts = line.split()
                    new_lines.append(f"{parts[0]} {parts[1]} {parts[2]} 1\n")
                elif line.startswith("v"):
                    parts = line.split()
                    new_lines.append(f"{parts[0]} {parts[1]} {parts[2]}\n")
                else:
                    new_lines.append(line+"\n")
        with open(os.path.join(file_path, f"{file_name}.lg"), "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    @staticmethod
    def exp_init(exp_name, exp_description):
        """Initialize experiment environment
        
        Args:
            exp_name: Experiment name
            exp_description: Experiment description
        
        Returns:
            list: List of completed supports
        """
        exp_dir = os.path.join(results_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        status_path = os.path.join(exp_dir, "experiment_status.json")
        if os.path.exists(status_path):
            with open(status_path, 'r', encoding='utf-8') as f:
                status = json.load(f)
                supports = status.get('completed_supports', [])
                supports.sort()
        else:
            status = {
                'name': exp_name,
                'description': exp_description,
                'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'completed_supports': [],
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(status_path, 'w') as f:
                json.dump(status, f, indent=2)
            supports = []
            
        return supports

    @staticmethod
    def run_grami(data_path, support):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        shell_path = os.path.join(current_dir, "GraMi-master", "run.sh")
        
        if not os.path.exists(shell_path):
            raise FileNotFoundError(f"Shell script not found at {shell_path}")
        
        try:
            print(f"Starting GraMi execution - Data file: {data_path}, Support: {support}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Script path: {shell_path}")
            
            process = subprocess.Popen(
                ["sh", shell_path, data_path, str(support)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=60)
                if process.returncode == 0:
                    print(f"GraMi execution successful: {stdout}")
                else:
                    print(f"GraMi execution failed: {stderr}")
                    raise subprocess.CalledProcessError(process.returncode, ["sh", shell_path, data_path, str(support)])
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    stdout, stderr = process.communicate(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                print("GraMi execution timed out (60 seconds)")
                raise
                
        except subprocess.CalledProcessError as e:
            print(f"GraMi execution failed: {e.stderr}")
            print(f"Return code: {e.returncode}")
            print(f"Command: {e.cmd}")
            raise
        except Exception as e:
            print(f"Unknown error during GraMi execution: {str(e)}")
            print(f"Error type: {type(e)}")
            raise
        
    def _save_patterns(self, data_path, model_name, result_dir):
        """Save mined frequent patterns"""
        shutil.copy(data_path, os.path.join(result_dir, model_name+".txt"))
        
    def _update_status(self, support):
        """Update experiment status"""
        status_path = os.path.join(results_dir, self.exp_name, "experiment_status.json")
        with open(status_path, 'r') as f:
            status = json.load(f)
        
        if support not in status['completed_supports']:
            status['completed_supports'].append(support)
            status['completed_supports'].sort()
            status['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        with open(status_path, 'w') as f:
            json.dump(status, f, indent=2)

    def run_experiment(self):
        global model_list
        self.initialize()
        for support_value in self.support_values:
            if support_value in self.completed_supports:
                print(f"Skipping completed experiment: support = {support_value}")
                continue
                
            print(f"Running experiment: support = {support_value}")
            result_dir = os.path.join(results_dir, self.exp_name, "frequent_patterns", f"support_{support_value}")
            os.makedirs(result_dir, exist_ok=True)
            
            for model_name in model_list:
                lg_data_path = model_name+".lg"
                self.run_grami(lg_data_path, support_value)
                self._save_patterns(lg_output_path, model_name, result_dir)
                
            self._update_status(support_value)

    def overview_analysis(self):
        """Overview analysis: Count frequent patterns for each model at different supports
        Generate Excel table for recording and updating
        """
        exp_dir = os.path.join(results_dir, self.exp_name, "frequent_patterns")
        if not os.path.exists(exp_dir):
            raise FileNotFoundError(f"Experiment directory does not exist: {exp_dir}")
        
        stats = {}
        
        for support_dir in os.listdir(exp_dir):
            if not support_dir.startswith('support_'):
                continue
                
            support = int(support_dir.split('_')[1])
            stats[support] = {}
            
            support_path = os.path.join(exp_dir, support_dir)
            for model_file in os.listdir(support_path):
                if not model_file.endswith('.txt'):
                    continue
                    
                model_name = model_file[:-4]  
                file_path = os.path.join(support_path, model_file)
                
                try:
                    with open(file_path, 'r') as f:
                        f.readline()
                        second_line = f.readline().strip()
                        pattern_count = int(second_line)
                        stats[support][model_name] = pattern_count
                except Exception as e:
                    print(f"Reading file failed {file_path}: {str(e)}")
                    stats[support][model_name] = None
        
        df = pd.DataFrame(stats).T  
        
        df = df.sort_index()
        
        excel_path = os.path.join(results_dir, self.exp_name, 'overview_pattern_counts.xlsx')
        
        if os.path.exists(excel_path):
            existing_df = pd.read_excel(excel_path, index_col=0)
            for support in stats:
                if support in existing_df.index:
                    for model in stats[support]:
                        if model in existing_df.columns:
                            existing_df.at[support, model] = stats[support][model]
            df = existing_df
        
        df.to_excel(excel_path)
        print(f"Frequent pattern count statistics saved to: {excel_path}")

    def _analyze_single_pattern(self, args):
        support, model_name, pattern_file, raw_data_file = args
        
        edge_count, _ = analyze_edge_and_path_metrics(pattern_file)
        
        avg_level, _, _, _ = analyze_pattern_metrics(
            raw_data_file, 
            pattern_file
        )
        
        return support, model_name, edge_count, avg_level

    def metric_analysis(self):
        exp_dir = os.path.join(results_dir, self.exp_name)
        if not os.path.exists(exp_dir):
            raise FileNotFoundError(f"Experiment directory does not exist: {exp_dir}")
        
        tasks = []
        for support_dir in os.listdir(os.path.join(exp_dir, "frequent_patterns")):
            if not support_dir.startswith('support_'):
                continue
            
            support = int(support_dir.split('_')[1])
            support_path = os.path.join(exp_dir, "frequent_patterns", support_dir)
            
            for model_file in os.listdir(support_path):
                print(support_path)
                if not model_file.endswith('.txt'):
                    continue
                
                model_name = model_file[:-4]
                pattern_file = os.path.join(support_path, model_file)
                raw_data_file = os.path.join(raw_data_dir, model_name + ".txt")
                
                tasks.append((support, model_name, pattern_file, raw_data_file))
        
        edge_stats = {}
        level_stats = {}
        
        max_workers = min(multiprocessing.cpu_count(), len(tasks))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(self._analyze_single_pattern, tasks):
                support, model_name, edge_count, avg_level = result
                
                for stats in [edge_stats, level_stats]:
                    if support not in stats:
                        stats[support] = {}
                
                edge_stats[support][model_name] = edge_count
                level_stats[support][model_name] = avg_level

        dfs = {
            '|C|': pd.DataFrame(edge_stats).T,
            'L(P)': pd.DataFrame(level_stats).T,
        }
        
        excel_path = os.path.join(exp_dir, f'analysis_results.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
            for sheet_name, df in dfs.items():
                df.sort_index().to_excel(writer, sheet_name=sheet_name)
        
        print(f"Metric analysis results saved to: {excel_path}")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def parse_args():
    parser = argparse.ArgumentParser(description='Frequent Pattern Mining Experiment')
    parser.add_argument('--exp_name', type=str, required=True,
                       help='Name of the experiment')
    parser.add_argument('--support_start', type=int, required=True,
                       help='Start value for support range')
    parser.add_argument('--support_end', type=int, required=True,
                       help='End value for support range (inclusive)')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--exp_num', type=str, required=True,
                       help='Experiment number')
    parser.add_argument('--benchmark_model', type=str, required=True,
                       help='Benchmark model name')
    parser.add_argument('--cluster_num', type=str, required=True,
                       help='Cluster number')
    parser.add_argument('--config', type=str, required=True, help='The path to the config file.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    raw_data_dir = "processed_data/{dataset}/exp{num}_results_{benchmark_model}/graph_cluster_{cluster_num}_dfs"
    raw_data_dir = raw_data_dir.format(
        dataset=args.dataset,
        num=args.exp_num,
        benchmark_model=args.benchmark_model,
        cluster_num=args.cluster_num
    )
    model_list = load_config(args.config).get("model_list", [])
    
    supports = [i for i in range(args.support_start, args.support_end + 1)]
    experiment = ExperimentManager(args.exp_name, supports)
    experiment.run_experiment()
    experiment.overview_analysis()
    experiment.metric_analysis()

