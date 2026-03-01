import argparse
import re
import json
import jsonlines
from datasets import load_from_disk
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
# from math_verify import parse, verify
from typing import Optional, Union, List
from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer
from openrlhf.utils.logging_utils import init_logger
# from openrlhf.utils.check.qwen_equal import math_equal
from multiprocessing import Pool
from transformers import AutoTokenizer
logger = init_logger(__name__)
import signal
from contextlib import contextmanager
import math
import torch
import torch.nn as nn
class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def math_equal2(gold, answer):
    try:
        gold=parse(gold)
        answer=parse(answer)
        return verify(gold, answer)
    except:
        return False
    
def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text
def extract_numbers(expr) -> list:
    curr_num = ''
    numbers = []
    for c in expr:
        if c.isdigit():
            curr_num += c
        elif curr_num:
            numbers.append(int(curr_num))
            curr_num = ''
    if curr_num:
        numbers.append(int(curr_num))
    return numbers
                
class RewardModelProxy:
    def __init__(self, args):
        self.reward_model = get_llm_for_sequence_regression(
            args.reward_pretrain,
            "reward",
            normalize_reward=args.normalize_reward,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            value_head_prefix=args.value_head_prefix,
            device_map="auto",
        )
        self.reward_model.eval()

        self.tokenizer = get_tokenizer(
            args.reward_pretrain, self.reward_model, "left", None, use_fast=not args.disable_fast_tokenizer
        )
        self.max_length = args.max_len
        self.batch_size = args.batch_size

    def get_reward(self, queries):
        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size

        # remove pad_token
        for i in range(len(queries)):
            queries[i] = (
                strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                + self.tokenizer.eos_token
            )
        logger.info(f"queries[0]: {queries[0]}")

        scores = []
        # batch
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                inputs = self.tokenize_fn(
                    queries[i : min(len(queries), i + batch_size)], device=self.reward_model.device
                )
                r = self.reward_model(inputs["input_ids"], inputs["attention_mask"])
                r = r.tolist()
                scores.extend(r)
        return scores

    def tokenize_fn(self, texts, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

class RuleTaj:
    """记录单轮规则使用情况和统计信息"""
    def __init__(self, rule_num: int, weights: List[float]):
        self.rule_num = rule_num  # 规则数量
        self.weights = weights.copy()  # 各规则权重
        self.usage = [0] * rule_num  # 各规则使用次数
        self.succ = [0] * rule_num   # 各规则成功次数

    def to_dict(self):
        """转换为字典用于历史记录"""
        return {
            "rule_num": self.rule_num,
            "weights": self.weights,
            "usage": self.usage,
            "succ": self.succ
        }
        

class RuleBasedRMProxy:
    def __init__(self, args):
        self.args = args
        self.prompt2answer = {}
        
        # dataset = load_from_disk(args.data_path)
        # train_list = list(dataset["train"])
        # validation_list = list(dataset["test"])
        
        # for line in train_list:
        #     self.prompt2answer[line['context'].strip()] = line['answer']
        # for line in validation_list:
        #     self.prompt2answer[line['context'].strip()] = line['answer']
        
        self.timeout_seconds=2
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        self.english_pattern = re.compile(r'[a-zA-Z]')
        self.boxed_pattern = re.compile(r"\\boxed\{((?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{(?:[^{}]|\\{|\\}|(?:\{[^{}]*\}))*\}))*\}))*\})")
        self.valid_char_pattern = re.compile(r'[a-zA-Z0-9\s\.,!?"\'\(\)\{\}\[\]_\-+=<>/@#$%^&*\\|:;~`\u2200-\u22FF]')
        self.repeat_pattern = re.compile(r'(.{5,}?)\1{4,}')
        self.lambda_pr = 0.75  # 步骤正确率
        self.lambda_l = 0.1  # 长度奖励系数
        self.lambda_oc = 0.15  # 结果奖励系数
        # self.lambda_pr = 0.3  # 步骤正确率
        # self.lambda_l = 0.15  # 长度奖励系数
        # self.lambda_oc = 0.55  # 结果奖励系数
        self.length_threshold = 4
        self.steepness = 0.2
        self.middle_offset = 20
        logger.info(f"步骤正确率： {self.lambda_pr}")
        logger.info(f"长度奖励系数： {self.lambda_l}")
        logger.info(f"结果奖励系数： {self.lambda_oc}")
        
        # 轮次计数器和权重更新周期
        self.current_round = 0
        self.weight_update_period = 5  # 每5轮更新一次权重
        logger.info(f"每{self.weight_update_period}轮更新一次权重")
        
        # 规则配置
        self.rule_num = 2  # 规则数量
        self.rule_weights = [1, 1]  # 初始规则权重
        self.hist = []  # 历史记录: 每轮的规则统计和奖励信息
        
    def set_rule_weights(self, new_weights: List[float]):
        """修改规则权重"""
        if len(new_weights) != self.rule_num:
            raise ValueError(f"需要{self.rule_num}个权重，实际提供{len(new_weights)}个")
        self.rule_weights = new_weights.copy()
        logger.info(f"规则权重已更新: {self.rule_weights}")
        
        
    def update_weights_based_on_history(self):
        """根据最近几轮的规则使用情况更新权重"""
        if len(self.hist) < self.weight_update_period:
            return  # 轮次不足，不更新
        
        # 获取最近weight_update_period轮的数据
        recent_history = self.hist[-self.weight_update_period:]
        
        # 计算每个规则的总使用率和成功率
        total_usage = [0] * self.rule_num
        total_success = [0] * self.rule_num
        
        for record in recent_history:
            stats = record["rule_stats"]
            for i in range(self.rule_num):
                total_usage[i] += stats["usage"][i]
                total_success[i] += stats["succ"][i]
        
        # 计算成功率（避免除零）
        success_rate = []
        for i in range(self.rule_num):
            if total_usage[i] == 0:
                success_rate.append(0.0)
            else:
                success_rate.append(total_success[i] / total_usage[i])
        
        # 计算使用率占比
        total_usage_sum = sum(total_usage)
        usage_ratio = []
        for i in range(self.rule_num):
            if total_usage_sum == 0:
                usage_ratio.append(1.0 / self.rule_num)  # 平均分配
            else:
                usage_ratio.append(total_usage[i] / total_usage_sum)
        
        # 计算使用率和成功率的综合得分
        # 可调整alpha参数来平衡使用率和成功率的重要性
        alpha = 0.5  # alpha越大，使用率越重要
        combined_scores = []
        for i in range(self.rule_num):
            score = alpha * usage_ratio[i] + (1 - alpha) * success_rate[i]
            combined_scores.append(score)
        
        # 基于综合得分计算新权重（归一化）
        sum_scores = sum(combined_scores)
        if sum_scores > 0:
            new_weights = [score / sum_scores for score in combined_scores]
        else:
            # 如果所有得分都是0，保持当前权重
            new_weights = self.rule_weights.copy()
        
        # 应用平滑因子，避免权重变化过大
        smoothing_factor = 0.7
        new_weights = [
            smoothing_factor * new_w + (1 - smoothing_factor) * old_w
            for new_w, old_w in zip(new_weights, self.rule_weights)
        ]
        
        # 确保权重总和为1
        weight_sum = sum(new_weights)
        if weight_sum > 0:
            new_weights = [w / weight_sum for w in new_weights]
        
        # 更新权重
        self.set_rule_weights(new_weights)
        
        # 记录权重更新日志
        logger.info(f"权重更新: 基于最近{self.weight_update_period}轮")
        logger.info(f"规则使用率: {total_usage}")
        logger.info(f"使用率占比: {usage_ratio}")
        logger.info(f"规则成功率: {success_rate}")
        logger.info(f"综合得分: {combined_scores}")
        logger.info(f"新权重: {new_weights}")
            
            
    def check_mixed_languages(self, text):
        chinese_chars = len(self.chinese_pattern.findall(text))
        english_chars = len(self.english_pattern.findall(text))
        return chinese_chars >= 20 and english_chars >= 20
    
    def check_garbled_characters(self, text):
        valid_chars = self.valid_char_pattern.sub('', text)
        if not text: 
            return False
        invalid_ratio = len(valid_chars) / len(text)
        return invalid_ratio > 0.3
    
    def has_repeated_patterns(self, text):
        return bool(self.repeat_pattern.search(text))
    
    def correctness_score(self, prompt, response):
        matches = self.boxed_pattern.findall(response)
        if not matches:
            return -1.0
        
        pred = matches[-1][:-1]
        if prompt not in self.prompt2answer:
            return -1.0
        
        return 1.0 if math_equal2(self.prompt2answer[prompt], pred) else -0.5
    
    def split_and_score(self, query):
        try:
            with timeout(self.timeout_seconds):
                if args.template_type=="qwen":
                    prompt=query.split("<|im_end|>\n<|im_start|>user\n")[-1].split("<|im_end|>\n<|im_start|>assistant\n")[0].strip()
                    response=query.split("<|im_end|>\n<|im_start|>assistant\n")[-1]
                    if "<|im_end|>" not in response and "<|endoftext|>" not in response:
                        return -1.0
                    response=query.split("<|im_end|>\n<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
                elif args.template_type=="deepseek":
                    prompt = query.split("<｜User｜>")[-1].split("<｜Assistant｜>")[0].strip()
                    prompt = prompt.replace("Please reason step by step, and put your final answer within \\boxed{}", "").strip()
                    response = query.split("<｜Assistant｜>")[-1].strip()
                    
                if "\\boxed" not in response or response.count("\\boxed")>=5:
                    return -1.0
                
                # if self.check_mixed_languages(response):
                #     return -1.0
                    
                if self.check_garbled_characters(response):
                    return -1.0
                
                if self.has_repeated_patterns(response):
                    return -1.0
                
                return self.correctness_score(prompt, response)
                
        except TimeoutException:
            logger.warning("Processing timed out")
            return -1.0
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return -1.0

    def length_reward_sigmoid(self, length):
        """S型递减的长度奖励"""
        if length <= self.length_threshold:
            return 1.0
        return 1 / (1 + math.exp(self.steepness * (length - self.length_threshold - self.middle_offset)))

    def rule_judge(self, num_a, op, num_b):
        """
        判断当前操作是否符合规则，返回每个规则的(使用次数, 成功次数)
        规则1: 乘法操作，且其中一个数能被24整除
        规则2: 加减操作，且两数差值小于3
        """
        try:
            a = int(num_a)
            b = int(num_b)
        except:
            return [(0, 0)] * self.rule_num
        
               
        # 规则1: 如果24能被任一操作数整除且操作符为乘法，则认为规则1被使用且成功
        if (24 % a == 0 or 24 % b == 0) and op == '*':
            rule_id = 0
            return rule_id
        # 规则2: 如果两个操作数的差的绝对值小于3且操作符为加减法，则认为规则2被使用且成功
        if abs(a - b) < 3 and op in ['+', '-']:
            rule_id = 1
            return rule_id
            
        return -1

    def check_solution(self, lines: List[str], print_error=False):

        """检查完整解决方案"""
        current_rule = RuleTaj(self.rule_num, self.rule_weights)
        """
        {
            "rule_num": self.rule_num,
            "weights": [],
            "usage": [],
            "succ": []
        }
        """
        # lines = text.strip().split('\n')
        # print('lines:',lines)
        start = lines[0]
        # print('start_init:',start)
        lines = lines[1:]
        error_lines = 0
        outcome_reward = 0
        rule_lines = 0
        rule_reward = 0
        # correct_reward = 0
        for i, line in enumerate(lines):

            flag = True
            if 'left:' in line:
                rule_lines +=1
                if 'roll' not in line and line in lines[:i]:  # 计算动作重复惩罚
                    error_lines += 1
                    continue
                exp_now = line.split(', left:')[0].strip()
                # print('exp_now', exp_now)

                try:
                    num_a, op, num_b = re.findall(r'\((\d+)\)\s*([\+\-\*/])\s*\((\d+)\)', exp_now)[0]
                    rule_id =  self.rule_judge(num_a,op,num_b)
                    if rule_id != -1:
                        current_rule.usage[rule_id] += 1
                    # print(current_rule)
                except Exception as e:
                    print('expr_now error!!!')
                    
                exprs = line.split('left:')[1].strip()
                if ',' in exprs or '=' in exprs:
                    exprs = exprs.split(',')
                else:
                    exprs = exprs.split(' ')
                numbers = []
                for expr in exprs:
                    if '=' in expr:
                        left_expr = expr.split('=')[0]
                        try:
                            right_value = eval(expr.split('=')[1])
                        except Exception as e:
                            error_lines += 1
                            flag = False
                            break
                        expr_numbers = extract_numbers(left_expr)
                        numbers += expr_numbers
                        try:
                            value = eval(left_expr)
                            if abs(value - right_value) > 1e-6:
                                error_lines += 1
                                flag = False
                                break
                        except Exception as e:
                            error_lines += 1
                            flag = False
                            break
                    else:
                        try:
                            value = int(expr)
                            numbers.append(value)
                        except Exception as e:
                            error_lines += 1
                            flag = False
                            break
            elif 'expression:' in line:
                if not i == len(lines) - 1:  # 冗余后缀惩罚
                    error_lines += 1
                    continue

                expression = line.split('expression:')[-1]
                numbers = extract_numbers(expression)
                try:
                    value = eval(expression)
                    if abs(value - 24) > 1e-6:  # 数值错误
                        error_lines += 1
                        flag = False
                except Exception as e:  # 表达式错误
                    flag = False
                    error_lines += 1
                print('start:',start)
                case = sorted([int(i) for i in start.split(' ')])
                numbers = sorted(numbers)
                if flag and numbers == case:
                    outcome_reward = 1
                    # suss = usage
                    current_rule.succ = current_rule.usage.copy()
                else: # 数字不匹配
                    error_lines += 1
                    outcome_reward = 0
                continue
            else:
                error_lines += 1
                continue
            case = sorted([int(i) for i in start.split(' ')])
            numbers = sorted(numbers)
            if flag and not numbers == case:
                error_lines += 1
                flag = False
            if not flag and print_error:
                print(f"Error in line : {line}")
                
        
        for i in range(current_rule.rule_num):
            rule_reward += current_rule.usage[i] * current_rule.weights[i]
            
            
        step_accuracy = 1 - error_lines / len(lines)
        # 记录本轮信息
        self.hist.append({
            "rule_stats": current_rule.to_dict(),
            "step_accuracy": step_accuracy,
            "outcome_reward": outcome_reward
        })
        print(self.current_round)
        print(current_rule.to_dict())
        return step_accuracy, rule_reward/rule_lines if rule_lines!=0 else 0,outcome_reward

    def evaluate_result(self, query):
        try:
            with timeout(self.timeout_seconds):
                if args.template_type == "qwen":
                    prompt = \
                    query.split("<|im_end|>\n<|im_start|>user\n")[-1].split("<|im_end|>\n<|im_start|>assistant\n")[
                        0].strip()
                    # print('prompt',prompt)
                    response = query.split("<|im_end|>\n<|im_start|>assistant\n")[-1]
                    # print('response',response)
                    if "<|im_end|>" not in response and "<|endoftext|>" not in response:
                        return -1.0
                    response = query.split("<|im_end|>\n<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].split(
                        "<|endoftext|>")[0].strip()
                elif args.template_type == "deepseek":
                    prompt = query.split("<｜User｜>")[-1].split("<｜Assistant｜>")[0].strip()
                    prompt = prompt.replace("Please reason step by step, and put your final answer within \\boxed{}",
                                            "").strip()
                    response = query.split("<｜Assistant｜>")[-1].strip()

        except TimeoutException:
            logger.warning("Processing timed out")
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")


        
        output = response
        # start = ' '.join([str(i) for i in prompt]).strip() + '\n'
        start = prompt.strip()+'\n'
        lines = (start + output).split('\n')

        # print('lines_init:',lines)
        reward_pr, reward_rul,reward_oc = self.check_solution(lines)
        reward_l = self.length_reward_sigmoid(len(lines) - 1)
        # reward = self.lambda_pr * (0.8*reward_pr+0.2*reward_rul) + self.lambda_l * reward_l + self.lambda_oc * reward_oc
        reward = self.lambda_pr * (0.6*reward_pr+0.4*reward_rul) + self.lambda_l * reward_l + self.lambda_oc * reward_oc  #grpo

       
        logger.info(f"reward_pr系数： 0.6")
        logger.info(f"reward_rul系数： 0.4")

        # reward = self.lambda_pr * reward_pr + self.lambda_l* reward_l + self.lambda_oc * reward_oc   #无规则


        # logger.info(f"reward_pr系数： 1")
        # logger.info(f"reward_rul系数： 0")
        # answers = {'reward_pr':reward_pr,'reward_rule':reward_rul,'reward_oc':reward_oc,'reward':reward}#记录变化
        #return reward
        self.current_round += 1
        # 权重调整
        # if self.current_round % self.weight_update_period == 0:
        #     self.update_weights_based_on_history()
        return reward
    
    def get_reward(self, queries):
        scores = []
        answer_l = []
        num_query = 0

        for query in queries:
            score = self.evaluate_result(query)
            # if answers == -1:
            #     scores.append(answers)
            #     continue
            # num_query += 1
            # print('answers',answers)
            # answer_l.append(answers)
            # score = answers['reward']
            scores.append(score)
        # if len(answer_l) != 0:
        #     for ans in answer_l:
        #         ans['num'] = len(answer_l)
        # file_name = 'reward_rule0603.json'
        # import os
        # if not os.path.exists('/data/coding/'+file_name):
        #     with open('/data/coding/'+file_name,'w') as file:
        #         json.dump(answer_l,file,indent=2,ensure_ascii=False)
        # else:
        #     with open('/data/coding/'+file_name,'r') as file:
        #         da = json.load(file)
        #     answer_l = da + answer_l
        #     with open('/data/coding/'+file_name,'w') as file:
        #         json.dump(answer_l,file,indent=2,ensure_ascii=False)
        #         print('saving...')
        



        return scores
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="rule")
    # RuleBasedRM Parameters
    parser.add_argument("--tokenizer_path", type=str, default='/data/coding/LLM4Game24/checkpoint/test_0409/checkpoint-1000')
    parser.add_argument("--max_gen_len", type=int)
    parser.add_argument("--template_type", type=str, default="qwen", choices=["qwen", "deepseek"])
    # Reward Model
    parser.add_argument("--data_path", type=str, default='/data/coding/data_train')    # for 
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--max_len", type=int, default="2048")

    parser.add_argument("--port", type=int, default=5061, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    # Performance
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=None)

    args = parser.parse_args()

    # server
    if args.mode=="model":
        reward_model = RewardModelProxy(args)
    else:
        reward_model = RuleBasedRMProxy(args)
    
    # test_case="<im_start>\nsystem\nnihao<|im_end|>\n<|im_start|>user\n1+1<|im_end|>\n<|im_start|>assistant\n1+1=\\boxed{2}<im_end>"
    # reward=reward_model.get_reward([test_case for _ in range(4)])
    # print(reward)
    # exit()
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        client_host = request.client.host
        logger.info(f"client_ip: {client_host}")
        data = await request.json()
        queries = data.get("query")
        rewards = reward_model.get_reward(queries)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
