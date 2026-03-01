import argparse
import re
import json
import numpy as np
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
                    queries[i: min(len(queries), i + batch_size)], device=self.reward_model.device
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
        self.succ = [0] * rule_num  # 各规则成功次数

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
        self.length_threshold = 4
        self.steepness = 0.2
        self.middle_offset = 20

        self.current_round = 0
        self.weight_update_period = 5
        logger.info(f"每{self.weight_update_period}轮更新一次权重")


        self.rule_num = 9
        self.rule_weights = [1/self.rule_num] * self.rule_num
        self.hist = []

    def set_rule_weights(self, new_weights: List[float]):

        if len(new_weights) != self.rule_num:
            raise ValueError(f"需要{self.rule_num}个权重，实际提供{len(new_weights)}个")
        self.rule_weights = new_weights.copy()
        logger.info(f"规则权重已更新: {self.rule_weights}")

    def update_weights_based_on_history(self):

        if len(self.hist) < self.weight_update_period:
            return


        recent_history = self.hist[-self.weight_update_period:]


        total_usage = [0] * self.rule_num
        total_success = [0] * self.rule_num

        for record in recent_history:
            stats = record["rule_stats"]
            for i in range(self.rule_num):
                total_usage[i] += stats["usage"][i]
                total_success[i] += stats["succ"][i]


        success_rate = []
        for i in range(self.rule_num):
            if total_usage[i] == 0:
                success_rate.append(0.0)
            else:
                success_rate.append(total_success[i] / total_usage[i])

        total_usage_sum = sum(total_usage)
        usage_ratio = []
        for i in range(self.rule_num):
            if total_usage_sum == 0:
                usage_ratio.append(1.0 / self.rule_num)
            else:
                usage_ratio.append(total_usage[i] / total_usage_sum)

        alpha = 0.5
        combined_scores = []
        for i in range(self.rule_num):
            score = alpha * usage_ratio[i] + (1 - alpha) * success_rate[i]
            combined_scores.append(score)

        sum_scores = sum(combined_scores)
        if sum_scores > 0:
            new_weights = [score / sum_scores for score in combined_scores]
        else:
            new_weights = self.rule_weights.copy()

        smoothing_factor = 0.7
        new_weights = [
            smoothing_factor * new_w + (1 - smoothing_factor) * old_w
            for new_w, old_w in zip(new_weights, self.rule_weights)
        ]

        weight_sum = sum(new_weights)
        if weight_sum > 0:
            new_weights = [w / weight_sum for w in new_weights]

        self.set_rule_weights(new_weights)



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
        if length <= self.length_threshold:
            return 1.0
        return 1 / (1 + math.exp(self.steepness * (length - self.length_threshold - self.middle_offset)))


    def calculate(self, a, op, b):
        if op == '+':
            return a + b
        elif op == '-':
            return a - b
        elif op == '*':
            return a * b
        elif op == '/':
            return a / b if b != 0 else 0
        return 0

    def rule_judge(self,num_a,op,num_b):
        rule = 0
        num_a = int(num_a)
        num_b = int(num_b)
        result = self.calculate(num_a, op, num_b)
        hit_condition = [0] * self.rule_num
        use_rule = [0] * self.rule_num

        hit_condition[0] += 1
        if result >= 1 and int(result) == result:
            rule += 1
            use_rule[0] = 1
        key_combinations = [(8,3), (6,4), (12,2), (4,4), (5,3)]
        if (num_a, num_b) in key_combinations or (num_b, num_a) in key_combinations:
            hit_condition[1] += 1
            if op in ['*', '+']:
                rule += 1
                use_rule[1] = 1
        if (24 % int(num_a) == 0 or 24 % int(num_b) == 0):
            hit_condition[2] += 1
            if op == '*':
                rule += 1
                use_rule[2] = 1
        hit_condition[3] += 1
        if result in [1, 2, 3, 4, 6, 8, 12]:
            rule += 1
            use_rule[3] = 1
        if abs(int(num_a)-int(num_b)) < 3:
            hit_condition[4] += 1
            if op in ['+','-']:
                rule += 1
                use_rule[4] = 1
        if abs(num_a - num_b) <= 1 and min(num_a, num_b) < 2:
            hit_condition[5] += 1
            if op == '+':
                rule += 1
                use_rule[5] = 1
        if (num_a not in [1,2,3,4,6,8,12,24] or num_b not in [1,2,3,4,6,8,12,24]):
            hit_condition[6] += 1
            if op in ['+', '-']:
                rule += 1
                use_rule[6] = 1
        if num_a + num_b in [6, 8, 10]:
            hit_condition[7] += 1
            if op == '+':
                rule += 1
                use_rule[7] = 1
        if abs(num_a - num_b) in [6, 8, 10]:
            hit_condition[8] += 1
            if op == '-':
                rule += 1
                use_rule[8] = 1
        return use_rule, hit_condition

    def check_solution(self, lines: List[str], print_error=False):

        current_rule = RuleTaj(self.rule_num, self.rule_weights)
        """
        {
            "rule_num": self.rule_num,
            "weights": [],
            "usage": [],
            "succ": []
        }
        """

        start = lines[0]
        lines = lines[1:]
        error_lines = 0
        outcome_reward = 0
        rule_lines = 0
        rule_reward = 0
        for i, line in enumerate(lines):

            flag = True
            if 'left:' in line:
                if 'roll' not in line and line in lines[:i]:
                    error_lines += 1
                    continue
                exp_now = line.split(', left:')[0].strip()

                try:
                    num_a, op, num_b = re.findall(r'\((\d+)\)\s*([\+\-\*/])\s*\((\d+)\)', exp_now)[0]
                    try:
                        succ_rule, hit_condition = self.rule_judge(num_a,op,num_b)
                        current_rule.usage = [x + y for x, y in zip(current_rule.usage, hit_condition)]
                        current_rule.succ = [x + y for x, y in zip(current_rule.succ, succ_rule)]
                        rule_lines += 1
                    except Exception as e:
                        pass
                except Exception as e:
                    rule_reward = rule_reward + 0
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
                if not i == len(lines) - 1:
                    error_lines += 1
                    continue

                expression = line.split('expression:')[-1]
                numbers = extract_numbers(expression)
                try:
                    value = eval(expression)
                    if abs(value - 24) > 1e-6:
                        error_lines += 1
                        flag = False
                except Exception as e:
                    flag = False
                    error_lines += 1
                print('start:',start)
                case = sorted([int(i) for i in start.split(' ')])
                numbers = sorted(numbers)
                if flag and numbers == case:
                    outcome_reward = 1
                    # suss = usage
                    # current_rule.succ = current_rule.usage.copy()
                else:
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
        
        current_reward_list = [x / y if y != 0 else 0 for x, y in zip(current_rule.succ, current_rule.usage)]

    
        for i in range(current_rule.rule_num):
            rule_reward += current_reward_list[i] * current_rule.weights[i]
        
        step_accuracy = 1 - error_lines / len(lines)

        self.hist.append({
            "rule_stats": current_rule.to_dict(),
            "step_accuracy": step_accuracy,
            "outcome_reward": outcome_reward
        })
        print(self.current_round)
        print(current_rule.to_dict())
        return step_accuracy, rule_reward, outcome_reward

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
        
        self.lambda_pr = 0.75
        self.lambda_l = 0.1
        self.lambda_oc = 0.15
        reward_rule = self.lambda_pr * (0.6*reward_pr+0.4*reward_rul) + self.lambda_l * reward_l + self.lambda_oc * reward_oc#grpo

        self.lambda_pr = 0.3
        self.lambda_l = 0.15
        self.lambda_oc = 0.55
        reward_no_rule = self.lambda_pr * reward_pr + self.lambda_l* reward_l + self.lambda_oc * reward_oc

        # reward = reward_rule
        reward = reward_no_rule

        answers = {'reward_pr':reward_pr,'reward_rule':reward_rul,'reward_oc':reward_oc,'reward':reward}#记录变化
        #return reward
        print(answers)
        print(f"reward: {reward}")
        print(f"reward_rule: {reward_rule}")
        print(f"reward_no_rule: {reward_no_rule}")
        # if reward_rul != 0 :
        #     print(f"reward_rul: {reward_rul}")
        self.current_round += 1
        if self.current_round % self.weight_update_period == 0:
            self.update_weights_based_on_history()
        return reward, reward_oc
    
    def get_reward(self, queries):
        scores = []
        out_scores = []
        answer_l = []
        num_query = 0

        for query in queries:
            score, out_score = self.evaluate_result(query)
            scores.append(score)
            out_scores.append(out_score)
        return scores, out_scores

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

def evaluate_result(test_cases, results):
    correct = 0
    error = 0
    unreach = 0
    correct_list = [0] * len(test_cases)
    for i,result in enumerate(results):
        lines = result.split('\n')
        last_line = lines[-1].strip().replace("<|im_end|>", "")
        if "expression:" not in last_line:
            unreach += 1
            continue
        expression = last_line.split('expression:')[-1]
        nums = extract_numbers(expression)
        if not test_cases[i].sort() == nums.sort():
            error += 1
            continue

        try:
            value = eval(expression)
            # print('ques:',test_cases[i])
            # print('final:',expression)
            if abs(value - 24) < 1e-6:
                correct += 1
                correct_list[i] = 1
            else:
                error += 1
        except Exception as e:
            error += 1
    return correct_list

class args:
    tokenizer_path = ''
    template_type = 'qwen'


reward_model = RuleBasedRMProxy(args)

def reward_func(queries, prompts, labels):
    prompt_queries = [prompt + query for prompt, query in zip(prompts, queries)]
    # print(json.dumps(prompt_queries, indent=2, ensure_ascii=False))
    scores, out_scores = reward_model.get_reward(prompt_queries)
    reward = torch.tensor(scores)
    inputs = [extract_numbers(prompt)[-4:] for prompt in prompts]
    print("inputs: ", inputs)
    acc_list = evaluate_result(inputs, queries)
    print("acc_list: ", acc_list)
    # acc_list = [0.5] * len(queries)
    acc = torch.tensor(acc_list)
    outcome_reward = torch.tensor(out_scores)
    # return reward
    return {
        "rewards": reward,  # Rewards for advantage calculation
        "scores": reward,  # Scores for dynamic filtering (0-1 reward)
        "extra_logs": {"dummy_scores": reward, "acc": acc, "outcome_reward": outcome_reward},  # Additional logging info for wandb
    }

