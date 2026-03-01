import argparse
import re
import json
import jsonlines
from datasets import load_from_disk,load_dataset
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
import re
from openai import OpenAI
import asyncio
from openai import AsyncOpenAI
import time
import nest_asyncio
import uvloop
# asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# nest_asyncio.apply()



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

# def math_equal2(gold, answer):
#     try:
#         gold=parse(gold)
#         answer=parse(answer)
#         return verify(gold, answer)
#     except:
#         return False
    
# def strip_sequence(text, pad_token, eos_token):
#     pad_token_escaped = re.escape(pad_token)
#     eos_token_escaped = re.escape(eos_token)

#     pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
#     text = re.sub(pattern, "", text)

#     pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
#     text = re.sub(pattern, "", text)
#     return text
# def extract_numbers(expr) -> list:
#     curr_num = ''
#     numbers = []
#     for c in expr:
#         if c.isdigit():
#             curr_num += c
#         elif curr_num:
#             numbers.append(int(curr_num))
#             curr_num = ''
#     if curr_num:
#         numbers.append(int(curr_num))
#     return numbers
                
# class RewardModelProxy:
#     def __init__(self, args):
#         self.reward_model = get_llm_for_sequence_regression(
#             args.reward_pretrain,
#             "reward",
#             normalize_reward=args.normalize_reward,
#             use_flash_attention_2=args.flash_attn,
#             bf16=args.bf16,
#             load_in_4bit=args.load_in_4bit,
#             value_head_prefix=args.value_head_prefix,
#             device_map="auto",
#         )
#         self.reward_model.eval()

#         self.tokenizer = get_tokenizer(
#             args.reward_pretrain, self.reward_model, "left", None, use_fast=not args.disable_fast_tokenizer
#         )
#         self.max_length = args.max_len
#         self.batch_size = args.batch_size

#     def get_reward(self, queries):
#         if self.batch_size is None:
#             batch_size = len(queries)
#         else:
#             batch_size = self.batch_size

#         # remove pad_token
#         for i in range(len(queries)):
#             queries[i] = (
#                 strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
#                 + self.tokenizer.eos_token
#             )
#         logger.info(f"queries[0]: {queries[0]}")

#         scores = []
#         # batch
#         with torch.no_grad():
#             for i in range(0, len(queries), batch_size):
#                 inputs = self.tokenize_fn(
#                     queries[i : min(len(queries), i + batch_size)], device=self.reward_model.device
#                 )
#                 r = self.reward_model(inputs["input_ids"], inputs["attention_mask"])
#                 r = r.tolist()
#                 scores.extend(r)
#         return scores

#     def tokenize_fn(self, texts, device):
#         batch = self.tokenizer(
#             texts,
#             return_tensors="pt",
#             add_special_tokens=False,
#             max_length=self.max_length,
#             padding=True,
#             truncation=True,
#         )
#         return {k: v.to(device) for k, v in batch.items()}




# def parse_pred(pred_str):
#     """最终修正版解析函数"""
#     parsed_steps = []
#     for line in pred_str.strip().split("\n"):
#         line = line.strip().lower()

#         # 预处理：移除所有非颜色词汇
#         line = re.sub(
#             r"\b(block|cube|on top of|the|a|an|from|of|on|onto)\b",
#             " ",
#             line
#         )
#         line = re.sub(r"\s+", " ", line).strip()  # 合并空格

#         # 核心解析逻辑
#         if "unstack" in line:
#             # 格式：unstack [颜色A] [颜色B]
#             parts = line.split()
#             if len(parts) >= 3 and parts[0] == "unstack":
#                 parsed_steps.append(f"(unstack {parts[1]} {parts[2]})")

#         elif "stack" in line:
#             # 格式：stack [颜色A] [颜色B]
#             parts = line.split()
#             if len(parts) >= 3 and parts[0] == "stack":
#                 parsed_steps.append(f"(stack {parts[1]} {parts[2]})")

#         elif "pick up" in line:
#             # 格式：pick up [颜色]
#             color = line.replace("pick up", "").strip()
#             if color:
#                 parsed_steps.append(f"(pick-up {color})")

#         elif "put down" in line:
#             # 格式：put down [颜色]
#             color = line.replace("put down", "").strip()
#             if color:
#                 parsed_steps.append(f"(put-down {color})")

#     return parsed_steps

ANSWER_START = "####"

# SYSTEM_PROMPT = (
#     "A conversation between User and Assistant. The user asks a question, and "
#     "the assistant solves it. The assistant first thinks about the reasoning "
#     "process in the mind and then provides the user with the answer. The "
#     "final answer is provided after the " + ANSWER_START + " tag, i.e., "
#     "{reasoning process} " + ANSWER_START + " {answer}."
# )

def delete_extra_zero(n):
    try:
        n=float(n)
    except:
        try:
            n = eval(n)
        except:
            # print("Conversion to floating number fails: {}".format(n))
            return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip("0")
        n = int(n.rstrip(".")) if n.endswith(".") else float(n)
        n=str(n)
        return n

def extract_from_response(text: str) -> str:
    try:
        answer = text.split(ANSWER_START)[-1].strip()
        if answer.endswith("."):
            answer = answer[:-1].strip()
        return answer
    except IndexError:
        return ""


def extract_hash_answer(text: str) -> str | None:
    try:
        return text.split("####")[1].strip()
    except IndexError:
        return None

def process_gsm8k_answer(pred: str) -> str:
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ")
    pred = [delete_extra_zero(s.replace(",", "")) 
            for s in re.findall(r"-?\d+/?\.?\d*", pred)]

    if len(pred) == 0:
        pred = ""
    else:
        pred = pred[-1].rstrip(".").rstrip("/")
    return pred




import re
from typing import List, Dict, Any

def extract_calculations(response_text: str, debug=False) -> List[Dict[str, Any]]:
    """
    修复版本：能处理各种数学表达式格式
    """
    if debug:
        print(f"输入文本: {repr(response_text[:200])}")
    
    calculations = []
    
    # 预处理：移除换行符，保持空格
    text = response_text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)  # 压缩多个空格为一个
    
    if debug:
        print(f"预处理后: {repr(text[:300])}")
    
    # 扩展的运算符模式
    operators_pattern = r'([+\-*/×÷x])'  # 添加了 'x' 作为乘号
    
    # 多种计算模式
    patterns = [
        # 模式1: 标准格式 "2 x 0.5 = 1"
        r'(\d+(?:\.\d+)?)\s*([+\-*/×÷x])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)',
        
        # 模式2: 带小数的格式 "2.5 + 1.0 = 3.5"
        r'(\d+(?:\.\d+)?)\s*([+\-*/×÷x])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)',
        
        # 模式3: 更宽松的格式，允许文字间隔
        r'(\d+(?:\.\d+)?)[^=\d]*([+\-*/×÷x])[^=\d]*(\d+(?:\.\d+)?)[^=\d]*=\s*(\d+(?:\.\d+)?)',
        
        # 模式4: 检测答案格式 "{数字}"
        r'answer is \{(\d+(?:\.\d+)?)\}',
        
        # 模式5: 简单的答案提取
        r'\{(\d+(?:\.\d+)?)\}',
    ]
    
    for i, pattern in enumerate(patterns):
        if debug:
            print(f"\n测试模式 {i+1}: {pattern}")
        
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        
        if debug:
            print(f"模式 {i+1} 找到 {len(matches)} 个匹配")
        
        for match in matches:
            if debug:
                print(f"  匹配内容: '{match.group(0)}'")
                print(f"  分组: {match.groups()}")
            
            # 根据模式类型解析
            if i < 3:  # 计算表达式模式
                calc_info = parse_calculation_expression(match, i)
            elif i == 3:  # "answer is {X}" 模式
                calc_info = {
                    'type': 'final_answer',
                    'result': float(match.group(1)),
                    'expression': match.group(0),
                    'position': match.start()
                }
            elif i == 4:  # "{X}" 模式
                calc_info = {
                    'type': 'answer_value',
                    'result': float(match.group(1)),
                    'expression': match.group(0),
                    'position': match.start()
                }
            
            if calc_info:
                calculations.append(calc_info)
    
    # 按位置排序
    calculations.sort(key=lambda x: x.get('position', 0))
    
    # 移除重复项
    calculations = remove_duplicates(calculations)
    
    if debug:
        print(f"\n最终提取到 {len(calculations)} 个计算:")
        for calc in calculations:
            print(f"  {calc}")
    
    return calculations

def parse_calculation_expression(match, pattern_id):
    """解析计算表达式"""
    groups = match.groups()
    
    if len(groups) >= 4:
        operand1 = float(groups[0])
        operator = normalize_operator(groups[1])
        operand2 = float(groups[2])
        result = float(groups[3])
        
        # 验证计算正确性
        expected = perform_calculation(operand1, operator, operand2)
        is_correct = abs(expected - result) < 1e-10
        
        return {
            'type': 'arithmetic',
            'expression': match.group(0),
            'operand1': operand1,
            'operator': operator,
            'operand2': operand2,
            'result': result,
            'expected_result': expected,
            'is_correct': is_correct,
            'position': match.start(),
            'pattern_id': pattern_id
        }
    
    return None

def normalize_operator(op: str) -> str:
    """标准化运算符"""
    op = op.strip().lower()
    operator_map = {
        '+': '+',
        '-': '-',
        '*': '*',
        '×': '*',
        'x': '*',  # 重要：将 'x' 识别为乘号
        '/': '/',
        '÷': '/',
    }
    return operator_map.get(op, op)

def perform_calculation(a: float, op: str, b: float) -> float:
    """执行计算"""
    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    elif op == '*':
        return a * b
    elif op == '/':
        return a / b if b != 0 else float('inf')
    else:
        raise ValueError(f"不支持的运算符: {op}")

def remove_duplicates(calculations: List[Dict]) -> List[Dict]:
    """移除重复计算"""
    seen = set()
    unique = []
    
    for calc in calculations:
        # 使用表达式和结果作为唯一标识
        if 'expression' in calc:
            key = calc['expression'].replace(' ', '').lower()
        else:
            key = f"{calc.get('result', '')}"
        
        if key not in seen:
            seen.add(key)
            unique.append(calc)
    
    return unique


# 验证函数
def verify_robe_calculations(calculations: List[Dict]) -> Dict[str, Any]:
    """验证长袍问题的计算"""
    verification = {
        'total_steps': len(calculations),
        'correct_steps': 0,
        'errors': [],
        'overall_correct': True
    }
    
    expected_calculations = [
        {'operand1': 2, 'operator': '*', 'operand2': 0.5, 'result': 1},
        {'operand1': 2, 'operator': '+', 'operand2': 1, 'result': 3}
    ]
    
    arithmetic_calcs = [c for c in calculations if c.get('type') == 'arithmetic']
    
    for i, calc in enumerate(arithmetic_calcs):
        if i < len(expected_calculations):
            expected = expected_calculations[i]
            if (calc['operand1'] == expected['operand1'] and 
                calc['operator'] == expected['operator'] and
                calc['operand2'] == expected['operand2'] and
                calc['result'] == expected['result']):
                verification['correct_steps'] += 1
            else:
                verification['errors'].append({
                    'step': i+1,
                    'expected': expected,
                    'actual': calc
                })
                verification['overall_correct'] = False
    
    # 检查最终答案
    final_answers = [c for c in calculations if c.get('type') in ['final_answer', 'answer_value']]
    if final_answers and final_answers[-1]['result'] == 3:
        verification['final_answer_correct'] = True
    else:
        verification['final_answer_correct'] = False
        verification['overall_correct'] = False
    
    return verification





import re
import math
from itertools import product

def word_to_number(text):
    """Convert word representation of numbers to numeric values."""
    # Dictionary mapping words to numbers
    word_map = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
        'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000,
        'million': 1000000, 'billion': 1000000000, 'trillion': 1000000000000
    }
    
    # Convert text to lowercase and handle hyphenated words
    text = text.lower().replace('-', ' ')
    
    # Find all words that represent numbers
    words = re.findall(r'\b(' + '|'.join(word_map.keys()) + r')\b', text)
    
    # Return the list of numeric values
    return [float(word_map[word]) for word in words]

def extract_numbers(text):
    """Extract all numbers from text, including decimals and word representations."""
    # Get numbers represented as digits
    digit_numbers = [float(n) for n in re.findall(r'\b\d+\.?\d*\b', text)]
    
    # Get numbers represented as words
    word_numbers = word_to_number(text)
    
    # Combine both lists
    return digit_numbers + word_numbers

def can_be_derived(num, source_numbers, depth=2):
    """Check if a number can be derived from source numbers with basic operations."""
    # Check if number is already in source numbers (with small epsilon for floating point)
    if any(abs(num - src) < 1e-6 for src in source_numbers):
        return True
    
    # Base case: if we've reached max depth, stop trying
    if depth <= 0:
        return False
    
    # Try all possible pairs of source numbers
    for a, b in product(source_numbers, repeat=2):
        # Try each arithmetic operation
        if abs(a + b - num) < 1e-6:
            return True
        if abs(a - b - num) < 1e-6:
            return True
        if abs(a * b - num) < 1e-6:
            return True
        if b != 0 and abs(a / b - num) < 1e-6:
            return True
    
    # If depth > 1, try creating intermediate results and check again
    if depth > 1:
        intermediate_results = set()
        for a, b in product(source_numbers, repeat=2):
            intermediate_results.add(a + b)
            intermediate_results.add(a - b)
            intermediate_results.add(a * b)
            if b != 0:
                intermediate_results.add(a / b)
        
        # Create new source list with intermediate results
        expanded_sources = list(source_numbers) + list(intermediate_results)
        
        # Recursive check with reduced depth
        return can_be_derived(num, expanded_sources, depth-1)
    
    return False

def analyze_reasoning(problem, reasoning):
    """Analyze if the reasoning introduces numbers not derivable from the problem."""
    problem_numbers = extract_numbers(problem)
    reasoning_numbers = extract_numbers(reasoning)
    
    invalid_numbers = []
    for num in reasoning_numbers:
        # Skip small integers that could be counting steps or items
        if num <= 5 and num.is_integer():
            continue
            
        if not can_be_derived(num, problem_numbers):
            invalid_numbers.append(num)
    
    return {
        "original_numbers": problem_numbers,
        "reasoning_numbers": reasoning_numbers,
        "invalid_numbers": invalid_numbers,
        "has_invalid_numbers": len(invalid_numbers) > 0
    }

def parse_conversation(text):
    """Extract problem and reasoning from conversation."""
    parts = text.split("assistant\n")
    if len(parts) < 2:
        return "", ""
    
    user_part = parts[0].split("user\n")[-1].strip()
    assistant_part = parts[1].strip()
    
    # Extract just the reasoning process
    if "{reasoning process}" in assistant_part:
        reasoning = assistant_part.split("{reasoning process}")[-1]
        if "####" in reasoning:
            reasoning = reasoning.split("####")[0]
    else:
        reasoning = assistant_part
        
    return user_part, reasoning

# Example usage
def analyze_conversation(conversation_text):
    problem, reasoning = parse_conversation(conversation_text)
    result = analyze_reasoning(problem, reasoning)
    

    
    return result



class RuleTaj:
    """记录单轮规则使用情况和统计信息"""
    def __init__(self, rule_num: int, weights: List[float]):
        self.rule_num = rule_num  # 规则数量
        self.weights = weights.copy()  # 各规则权重
        self.usage = [0] * rule_num  # 各规则使用次数
        self.succ = [0] * rule_num  # 各规则成功次数

    def to_dict(self) -> dict:
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
        
        self.timeout_seconds= 3
        # self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        
        # self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        # self.english_pattern = re.compile(r'[a-zA-Z]')
        # self.answer_pattern = re.compile(r"\[Solution\]\s*(.*)", re.DOTALL)
        # self.valid_char_pattern = re.compile(r'[a-zA-Z0-9\s\.,!?"\'\(\)\{\}\[\]_\-+=<>/@#$%^&*\\|:;~`\u2200-\u22FF]')
        # self.repeat_pattern = re.compile(r'(.{5,}?)\1{4,}')
        # self.lambda_pr = 0.75  # 步骤正确率
        # self.lambda_l = 0.1  # 长度奖励系数
        # self.lambda_oc = 0.15  # 结果奖励系数
        # self.length_threshold = 4
        # self.steepness = 0.2
        # self.middle_offset = 20
        self.prompt2answer = {}

        dataset = load_dataset(args.data_path)
        train_list = list(dataset['train'])
        for line in train_list:
            self.prompt2answer[line['question'].strip()] = line['answer'].strip()
            # print('context:', line['question'].strip())
            # print('answer:', self.prompt2answer[line['question'].strip()])


                # 轮次计数器和权重更新周期
        self.current_round = 0
        self.weight_update_period = 5  # 每5轮更新一次权重

        self.current_same = 0
        
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
        
        # 基于成功率计算新权重（归一化）
        new_weights = []
        sum_success = sum(success_rate)
        
        if sum_success > 0:
            new_weights = [rate / sum_success for rate in success_rate]
        else:
            # 如果所有成功率都是0，保持当前权重
            new_weights = self.rule_weights.copy()
        
        # 更新权重
        self.set_rule_weights(new_weights)
        
        # 记录权重更新日志
        logger.info(f"权重更新: 基于最近{self.weight_update_period}轮")
        logger.info(f"规则使用率: {total_usage}")
        logger.info(f"规则成功率: {success_rate}")
        logger.info(f"新权重: {new_weights}")
    

    def rule_verified_reward(self,prompt,response):
    # cals = extract_calculations(response)
        hit_condition = [0] * self.rule_num
        use_rule = [0] * self.rule_num


        expressions = extract_calculations(response)
        errors_exp = 0
        exp_score = 0
        judge_result = analyze_conversation(prompt+response)
        # has_errors = analyze_conversation(prompt+response)
        if judge_result['has_invalid_numbers']:
            number_valid = 1 - len(judge_result['invalid_numbers']) / len(judge_result['reasoning_numbers'])

            use_rule[0]+= len(judge_result['invalid_numbers'])
            hit_condition[0] += len(judge_result['reasoning_numbers'])
        else:
            number_valid = 1
            hit_condition[0] += len(judge_result['reasoning_numbers'])
        
        for exp in expressions:
            hit_condition[1] += 1
            try:
                if not exp['is_correct']:
                    errors_exp = errors_exp+1
                    use_rule[1] += 1
            except Exception as e:
                exp_score = 1
        if len(expressions) != 0:
            exp_score = 1-errors_exp / len(expressions)

        return use_rule, hit_condition
        # return 0.5*exp_score + 0.5*number_valid

        
    def rule_reward(self,prompt,response):
        # client = OpenAI(base_url="https://yunwu.ai/v1",api_key='sk-SwvzH9pMnRS6bxlpeX9j2dTOG89ZWzWUBHao9tiBBn7phlq2',)
        client = OpenAI(base_url="https://xiaoai.plus/v1",api_key='sk-I6Ko4OUb9ZrNpa144KRzaF5KkC8vBDGMW9fT4v3m27t9Y7uU',)
        re_prompt = reward_prompt.replace('{conversation context & query}',prompt).replace('{the i-th response}',response)
        try:
         res = client.chat.completions.create(model='gpt-4o-mini', messages=[{"role": "user", "content": re_prompt}])
        except Exception as e:
            print('chat error !!!')
            res = client.chat.completions.create(model='gpt-4o-mini', messages=[{"role": "user", "content": re_prompt}])
        pattern = r'\\boxed\{([^}]*)\}'
        match = re.search(pattern, res.choices[0].message.content.strip())
        if match:
            score = eval(match.group(1)) / 10
        else:
            # score = res.choices[0].text.strip()
            # print('no number!!!')
            res = client.chat.completions.create(model='gpt-4o-mini', messages=[{"role": "user", "content": re_prompt}])
            match = re.search(pattern, res.choices[0].message.content.strip())
            if match:
                score = eval(match.group(1)) / 10
            else:
                # print('final no number!!!')
                score = 1.0
        return score

        
        

        # except Exception as e:
            # print(e)
            




    def correctness_gsm8k_score(self,prompt,response):
        current_rule = RuleTaj(self.rule_num, self.rule_weights)

        succ_rule, hit_condition = self.rule_verified_reward(prompt,response)

        # succ_rule, hit_condition = self.rule_check(action_chain=pred,init_state=init_state,goal_state=goal_state)
        current_rule.usage = [x  for x in hit_condition]
        current_rule.succ = [x  for x in succ_rule]

        current_reward_list =  [x / y if y != 0 else 0 for x, y in zip(current_rule.succ, current_rule.usage)]
        error_rule = 0
        for i in range(current_rule.rule_num):
            error_rule += current_reward_list[i] * current_rule.weights[i]
         
        rule_score = 1.0 - error_rule

        extra_answer = extract_from_response(response)
        gen_ans = process_gsm8k_answer(extra_answer)
        true_answer = extract_hash_answer(self.prompt2answer[prompt])
        # true_answer = extract_hash_answer(dataset['answer'][i])
        true_answer = process_gsm8k_answer(true_answer)
        if gen_ans == true_answer:
            outcome_reward = 1.0
        else:
            outcome_reward = 0.0
        return 0.4*rule_score+0.6*outcome_reward

    #todo: rewrite correctness_score
    # def correctness_score(self, prompt, response):
    #     # print('response:',response)
    #     # print('prompt:',prompt)
    #     init_state,goal_state = self.extract_states(prompt)
    #     # goal = prompt.split('My goal is to have that')[1].split('My plan')[0].strip()
    #     # init = prompt.split('As initial conditions I have that,')[1].split("My goal")[0].strip()
    #     rule_score = 0
    #     try:
    #         # pred = response.split('[PLAN]')[1].split('[PLAN END]')[0]
    #         pred = response.split('[PLAN END]')[0]
    #     except Exception as e:
    #         return rule_score,0.0
    #     pred = parse_pred(pred)
    #     rule_score = self.rule_check(action_chain=pred,init_state=init_state,goal_state=goal_state)
    #     error_lines = 0
    #     truth = self.prompt2answer[prompt]
    #     if len(truth)!= len(pred):
    #         return rule_score,0.0

    #     for i in range(0,len(truth)):
    #         try:
    #             if truth[i] != pred[i]:
    #                 print('truth:',truth[i])
    #                 print('pred:',pred[i])
    #                 error_lines += 1
    #                 return rule_score,0.0
    #         except Exception as e:
    #             return rule_score,0.0
    #     # return -0.5*error_lines/float(len(truth)) if error_lines > 0 else 1.0
    #     return rule_score,1.0






        #
        # matches = self.answer_pattern.search(response)
        # if not matches:
        #     return -1.0
        #
        # pred = matches.group(1)
        # if prompt not in self.prompt2answer:
        #     return -1.0
        #
        # return 1.0 if pred in self.prompt2answer[prompt] else -0.5
    
    # def split_and_score(self, query):
    #     try:
    #         with timeout(self.timeout_seconds):
    #             if args.template_type=="qwen":
    #                 prompt=query.split("<|im_end|>\n<|im_start|>user\n")[-1].split("<|im_end|>\n<|im_start|>assistant\n")[0].strip()
    #                 response=query.split("<|im_end|>\n<|im_start|>assistant\n")[-1]
    #                 if "<|im_end|>" not in response and "<|endoftext|>" not in response:
    #                     return -1.0
    #                 response=query.split("<|im_end|>\n<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
    #             elif args.template_type=="deepseek":
    #                 prompt = query.split("<｜User｜>")[-1].split("<｜Assistant｜>")[0].strip()
    #                 prompt = prompt.replace("Please reason step by step, and put your final answer within \\boxed{}", "").strip()
    #                 response = query.split("<｜Assistant｜>")[-1].strip()

    #             if "\\boxed" not in response or response.count("\\boxed")>=5:
    #                 return -1.0

    #             # if self.check_mixed_languages(response):
    #             #     return -1.0

    #             if self.check_garbled_characters(response):
    #                 return -1.0

    #             if self.has_repeated_patterns(response):
    #                 return -1.0

    #             return self.correctness_score(prompt, response)
                
    #     except TimeoutException:
    #         logger.warning("Processing timed out")
    #         return -1.0
    #     except Exception as e:
    #         logger.error(f"Error processing query: {str(e)}")
    #         return -1.0

    # def length_reward_sigmoid(self, length):
    #     """S型递减的长度奖励"""
    #     if length <= self.length_threshold:
    #         return 1.0
    #     return 1 / (1 + math.exp(self.steepness * (length - self.length_threshold - self.middle_offset)))

    # def rule_judge(self,num_a,op,num_b):
    #     rule = 0
    #     if (24 % int(num_a) == 0 or 24 % int(num_b) == 0) and op == '*':
    #         rule = 1
    #         return rule
    #     if abs(int(num_a)-int(num_b)) < 3 and op in ['+','-']:
    #         rule = 1
    #         return rule
    #     # if num_a > 10 and num_b > 10 :
    #     #     rule -=1
    #     # if is_prime(num_a) and is_prime(num_b):
    #     #     rule -=1
    #     return rule

    # def check_solution(self, lines: List[str], print_error=False):

    #     """检查完整解决方案"""

    #     # lines = text.strip().split('\n')
    #     print('lines:',lines)
    #     start = lines[0]
    #     print('start_init:',start)
    #     lines = lines[1:]
    #     error_lines = 0
    #     outcome_reward = 0
    #     rule_lines = 0
    #     rule_reward = 0
    #     # correct_reward = 0
    #     for i, line in enumerate(lines):

    #         flag = True
    #         if 'left:' in line:
    #             rule_lines +=1
    #             if 'roll' not in line and line in lines[:i]:  # 计算动作重复惩罚
    #                 error_lines += 1
    #                 continue
    #             exp_now = line.split(', left:')[0].strip()
    #             # print('exp_now', exp_now)

    #             try:
    #                 num_a, op, num_b = re.findall(r'\((\d+)\)\s*([\+\-\*/])\s*\((\d+)\)', exp_now)[0]
    #                 rule_reward = rule_reward + self.rule_judge(num_a,op,num_b)
    #             except Exception as e:
    #                 print('expr_now error!!!',num_a,op,num_b)
    #                 rule_reward = 0
    #             exprs = line.split('left:')[1].strip()
    #             if ',' in exprs or '=' in exprs:
    #                 exprs = exprs.split(',')
    #             else:
    #                 exprs = exprs.split(' ')
    #             numbers = []
    #             for expr in exprs:
    #                 if '=' in expr:
    #                     left_expr = expr.split('=')[0]
    #                     try:
    #                         right_value = eval(expr.split('=')[1])
    #                     except Exception as e:
    #                         error_lines += 1
    #                         flag = False
    #                         break
    #                     expr_numbers = extract_numbers(left_expr)
    #                     numbers += expr_numbers
    #                     try:
    #                         value = eval(left_expr)
    #                         if abs(value - right_value) > 1e-6:
    #                             error_lines += 1
    #                             flag = False
    #                             break
    #                     except Exception as e:
    #                         error_lines += 1
    #                         flag = False
    #                         break
    #                 else:
    #                     try:
    #                         value = int(expr)
    #                         numbers.append(value)
    #                     except Exception as e:
    #                         error_lines += 1
    #                         flag = False
    #                         break
    #         elif 'expression:' in line:
    #             if not i == len(lines) - 1:  # 冗余后缀惩罚
    #                 error_lines += 1
    #                 continue

    #             expression = line.split('expression:')[-1]
    #             numbers = extract_numbers(expression)
    #             try:
    #                 value = eval(expression)
    #                 if abs(value - 24) > 1e-6:  # 数值错误
    #                     error_lines += 1
    #                     flag = False
    #             except Exception as e:  # 表达式错误
    #                 flag = False
    #                 error_lines += 1
    #             print('start:',start)
    #             case = sorted([int(i) for i in start.split(' ')])
    #             numbers = sorted(numbers)
    #             if flag and numbers == case:
    #                 outcome_reward = 1
    #             else: # 数字不匹配
    #                 error_lines += 1
    #                 outcome_reward = 0
    #             continue
    #         else:
    #             error_lines += 1
    #             continue
    #         case = sorted([int(i) for i in start.split(' ')])
    #         numbers = sorted(numbers)
    #         if flag and not numbers == case:
    #             error_lines += 1
    #             flag = False
    #         if not flag and print_error:
    #             print(f"Error in line : {line}")
    #     return 1 - error_lines / len(lines),rule_reward/rule_lines,outcome_reward

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
        # start = prompt.strip()+'\n'
        # lines = (start + output).split('\n')
        reward = self.correctness_gsm8k_score(prompt, output)
        self.current_round += 1

        logger.info(f"current_round数： {self.current_round}")
        # 权重调整
        if self.current_round % self.weight_update_period == 0:
            self.update_weights_based_on_history()
        # print('rule:',rule)
        # print('reward:',reward)

        return reward
        # return (rule,reward)
        # return reward

        # print('lines_init:',lines)
        # reward_pr, reward_rul,reward_oc = self.check_solution(lines)
        # reward_l = self.length_reward_sigmoid(len(lines) - 1)
        # reward = self.lambda_pr * (0.8*reward_pr+0.2*reward_rul) + self.lambda_l * reward_l + self.lambda_oc * reward_oc
        # # reward = self.lambda_pr * reward_pr + self.lambda_l* reward_l + self.lambda_oc * reward_oc
        #
        # return reward
    
    def split_prompt_response(self,queries):
        results = {}
        prompts = [] 
        responses = []
        scores = []
        for q in queries:
            try:
                with timeout(self.timeout_seconds):
                 if args.template_type == "qwen":
                    prompt = \
                    query.split("<|im_end|>\n<|im_start|>user\n")[-1].split("<|im_end|>\n<|im_start|>assistant\n")[
                        0].strip()
                    # print('prompt',prompt)
                    score = 0
                    response = query.split("<|im_end|>\n<|im_start|>assistant\n")[-1]
                    prompts.append(prompt)
                    responses.append(response)
                    scores.append(score)
                    # print('response',response)
                    if "<|im_end|>" not in response and "<|endoftext|>" not in response:
                        # score =  -1.0
                        scores[-1] = -1.0
                        continue
                    response = query.split("<|im_end|>\n<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].split(
                        "<|endoftext|>")[0].strip()
                    responses[-1] = response
            except TimeoutException:
                logger.warning("Processing timed out")
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
        results['prompt'] = prompts
        results['response'] = responses
        results['score'] = scores
        return results
    def evaluate_result_batch(self,prompt,response,score,rule):
        if all_result['score'] == -1.0:
            return -1.0
        reward = self.correctness_gsm8k_score(all_result['prompt'],all_result['response'])
        
        return 0.2*rule+ 0.8*reward

        



    def get_reward(self, queries):
        scores = []
        rules = []
        for query in queries:
            score = self.evaluate_result(query)
            # rules.append(rule)
            scores.append(score)
        # print('rule average:',sum(rules)/len(rules))
        print('score average:',sum(scores)/len(scores))

        return scores
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="rule")
    # RuleBasedRM Parameters
    parser.add_argument("--tokenizer_path", type=str, default='/data/coding/Qwen2.5-0.5B-Instruct')
    parser.add_argument("--max_gen_len", type=int)
    parser.add_argument("--template_type", type=str, default="qwen", choices=["qwen", "deepseek","llama"])
    # Reward Model
    parser.add_argument("--data_path", type=str, default='/data/coding/gsm8k')    # for
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--max_len", type=int, default="2048")

    parser.add_argument("--port", type=int, default=5001, help="Port number for the server")
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
        # logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
