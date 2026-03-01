import prompts
import pandas as pd
import os
from openai import OpenAI
import argparse
import re
from tqdm import tqdm
import time
from prompts import ZERO_SHOT_FORMAT_TEMPLATE, FEW_SHOTS_FORMAT_TEMPLATE, OURS_FORMAT_TEMPLATE

class ResponseFormatter:
    def __init__(self, model, max_retries=3, type="zero_shot"):
        self.model = model
        self.max_retries = max_retries
        self.type = type
        self.validator = ResponseValidator()
    
    def format_response(self, original_response):
        """Format response to ensure it meets standard format"""
        response = original_response
        retry_count = 0
        
        while retry_count < self.max_retries:
            if self.is_well_formatted(response) and self.validator.validate_format(response):
                return response
                
            if self.type == "zero_shot":
                prompt = ZERO_SHOT_FORMAT_TEMPLATE.format(prev_response=response)
            elif self.type == "few_shots":
                prompt = FEW_SHOTS_FORMAT_TEMPLATE.format(prev_response=response)
            elif self.type == "ours":
                prompt = OURS_FORMAT_TEMPLATE.format(prev_response=response)
            else:
                raise ValueError(f"Invalid type: {self.type}")
            try:
                new_response = self.model.chat.completions.create(
                    model = args.base_lm,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.01,
                    max_tokens=1024,
                )
                response = new_response.choices[0].message.content
            except Exception as e:
                print(f"Format error: {str(e)}")
            
            retry_count += 1
            
        return response
    
    def is_well_formatted(self, response):
        """Check if response meets basic format requirements"""
        if not response:
            return False
        
        if "Answer:" not in response:
            return False
            
        parts = response.split("Answer:")
        if len(parts) < 2:
            return False
            
        answer_part = parts[-1].strip()
        if "=" not in answer_part:
            return False
            
        if not answer_part.strip().endswith("= 24"):
            return False
            
        return True

class ResponseValidator:
    @staticmethod
    def validate_format(response):
        """Validate the legality of response format"""
        try:
            parts = response.split("Answer:")
            if len(parts) < 2:
                return False
                
            reasoning = parts[0].strip()
            answer = parts[1].strip()
            
            if "=" not in answer:
                return False
                
            equations = answer.split("=")
            if len(equations) < 2:
                return False
                
            expression = equations[-2].strip()
            result = equations[-1].strip()
            
            if result != "24":
                return False
                
            if not re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', expression):
                return False
                
            return True
            
        except Exception:
            return False

def extract_answer(response):
    """Extract reasoning process and answer from correctly formatted response"""
    parts = response.split("Answer:")
    reasoning = parts[0].strip().replace("\n", " ")
    answer = parts[1].strip().replace("\n", " ")
    return reasoning, answer

def validate_answer(answer):
    """Validate the correctness of the answer"""
    try:
        equations = answer.split("=")
        formula = equations[-2].strip()
        result = float(eval(formula))
        return abs(result - 24) < 1e-6
    except:
        return False

def log_result(f, case_id, question, reasoning, answer, is_correct, correct_count, total_count):
    accuracy = correct_count / total_count
    log_line = f'Case #{case_id}: question="{question}", reasoning="{reasoning}", output=\'{answer}\', answer=\'{answer}\';accuracy={accuracy:.3f} ({correct_count}/{total_count})\n'
    f.write(log_line)
    f.flush()
    
def update_global_performance(model_name, accuracy, avg_prompt_tokens, avg_completion_tokens, avg_time, global_csv_path="global_performance.csv"):
    """Update global performance records
    
    Args:
        model_name: Model name
        accuracy: Current experiment accuracy
        global_csv_path: Global record file path
    """
    try:
        if os.path.exists(global_csv_path):
            df = pd.read_csv(global_csv_path)
        else:
            df = pd.DataFrame(columns=["model", "accuracy", "avg_prompt_tokens", 
                                     "avg_completion_tokens", "avg_time", "experiment_count"])
        
        if model_name in df["model"].values:
            idx = df.index[df["model"] == model_name].item()
            exp_count = df.at[idx, "experiment_count"] + 1
            
            df.at[idx, "accuracy"] = (df.at[idx, "accuracy"] * (exp_count-1) + accuracy) / exp_count
            df.at[idx, "avg_prompt_tokens"] = (df.at[idx, "avg_prompt_tokens"] * (exp_count-1) + avg_prompt_tokens) / exp_count
            df.at[idx, "avg_completion_tokens"] = (df.at[idx, "avg_completion_tokens"] * (exp_count-1) + avg_completion_tokens) / exp_count
            df.at[idx, "avg_time"] = (df.at[idx, "avg_time"] * (exp_count-1) + avg_time) / exp_count
            df.at[idx, "experiment_count"] = exp_count
            
        else:
            new_row = pd.DataFrame({
                "model": [model_name],
                "accuracy": [accuracy],
                "avg_prompt_tokens": [avg_prompt_tokens],
                "avg_completion_tokens": [avg_completion_tokens], 
                "avg_time": [avg_time],
                "experiment_count": [1]
            })
            df = pd.concat([df, new_row], ignore_index=True)
            
        df.to_csv(global_csv_path, index=False)
        print(f"Updated performance record for {model_name}")
            
    except Exception as e:
        print(f"Error updating global performance record: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_lm", type=str, default="glm-4-9b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--mode", type=str, default="zero_shot")
    args = parser.parse_args()

    model = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )
    prompt_template = prompts.zero_shot
    if args.mode == "zero_shot":
        prompt_template = prompts.zero_shot
    elif args.mode == "few_shots":
        prompt_template = prompts.few_shots
    elif args.mode == "ours":
        if "gpt" in args.base_lm:
            prompt_template = prompts.ours_v4
        else:
            prompt_template = prompts.ours_v3
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    data_path = "case_study/rule_patterns/game24/24.csv"
    data = pd.read_csv(data_path)[["Puzzles"]][900:1000]
    os.makedirs(args.log_dir, exist_ok=True)
    
    config_log_path = os.path.join(args.log_dir, f"{args.mode}_{args.base_lm}_config.log")
    results_log_path = os.path.join(args.log_dir, f"{args.mode}_{args.base_lm}_results.log")
    
    with open(config_log_path, "w") as f:
        f.write("=== Experiment Configuration ===\n")
        f.write(f"Time: {pd.Timestamp.now()}\n")
        f.write(f"Model: {args.base_lm}\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Data Range: {data.index.min()}-{data.index.max()}\n")
        f.write(f"Sample Size: {len(data)}\n")
        f.write("===========================\n")
        f.flush()
    
    correct_count = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_time = 0
    
    formatter = ResponseFormatter(model, type=args.mode)
    
    with open(results_log_path, "w") as f:
        id = 0
        for question in tqdm(data["Puzzles"], desc="Processing {}".format(args.base_lm)):
            start_time = time.time()  # Start timing
            
            is_correct = False
            reasoning = ""
            answer = None
            
            try:
                prompt = prompt_template.replace("{QUESTION}", question)
                response = model.chat.completions.create(
                    model=args.base_lm,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=args.temperature,
                    max_tokens=1024,
                )
                original_response = response.choices[0].message.content
                print(original_response)
                
                total_prompt_tokens += response.usage.prompt_tokens
                total_completion_tokens += response.usage.completion_tokens
                
                formatted_response = formatter.format_response(original_response)
                
                if formatted_response:
                    reasoning, answer = extract_answer(formatted_response)
                    is_correct = validate_answer(answer)
                    if is_correct:
                        correct_count += 1
                
            except Exception as e:
                print(f"{args.base_lm} error processing question {id+1}: {str(e)}")
            
            total_time += time.time() - start_time  # Record time spent
            
            log_result(f, id+900, question, reasoning, answer, is_correct, correct_count, id+1)
            
            id += 1
            

    avg_prompt_tokens = total_prompt_tokens / len(data)
    avg_completion_tokens = total_completion_tokens / len(data)
    avg_time = total_time / len(data)

    final_accuracy = correct_count / len(data)
    global_csv_path = "case_study/rule_patterns/game24/logs/global_performance_{}.csv".format(args.mode)
    update_global_performance(
        args.base_lm, 
        final_accuracy,
        avg_prompt_tokens,
        avg_completion_tokens,
        avg_time,
        global_csv_path
    )







