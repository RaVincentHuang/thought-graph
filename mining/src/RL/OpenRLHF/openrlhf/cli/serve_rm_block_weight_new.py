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

def parse_pred(pred_str):
    """最终修正版解析函数"""
    parsed_steps = []
    for line in pred_str.strip().split("\n"):
        line = line.strip().lower()

        # 预处理：移除所有非颜色词汇
        line = re.sub(
            r"\b(block|cube|on top of|the|a|an|from|of|on|onto)\b",
            " ",
            line
        )
        line = re.sub(r"\s+", " ", line).strip()  # 合并空格

        # 核心解析逻辑
        if "unstack" in line:
            # 格式：unstack [颜色A] [颜色B]
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "unstack":
                parsed_steps.append(f"(unstack {parts[1]} {parts[2]})")

        elif "stack" in line:
            # 格式：stack [颜色A] [颜色B]
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "stack":
                parsed_steps.append(f"(stack {parts[1]} {parts[2]})")

        elif "pick up" in line:
            # 格式：pick up [颜色]
            color = line.replace("pick up", "").strip()
            if color:
                parsed_steps.append(f"(pick-up {color})")

        elif "put down" in line:
            # 格式：put down [颜色]
            color = line.replace("put down", "").strip()
            if color:
                parsed_steps.append(f"(put-down {color})")

    return parsed_steps

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
        self.answer_pattern = re.compile(r"\[Solution\]\s*(.*)", re.DOTALL)
        self.valid_char_pattern = re.compile(r'[a-zA-Z0-9\s\.,!?"\'\(\)\{\}\[\]_\-+=<>/@#$%^&*\\|:;~`\u2200-\u22FF]')
        self.repeat_pattern = re.compile(r'(.{5,}?)\1{4,}')
        self.lambda_pr = 0.75  # 步骤正确率
        self.lambda_l = 0.1  # 长度奖励系数
        self.lambda_oc = 0.15  # 结果奖励系数
        self.length_threshold = 4
        self.steepness = 0.2
        self.middle_offset = 20
        self.prompt2answer = {}

        # 动态权重相关配置
        self.rule_num = 2 # 规则数量： 5-规则
        self.initial_weights = [1/self.rule_num] * self.rule_num  # 初始权重
        self.rule_weights = self.initial_weights.copy()  # 当前权重
        self.hist = []  # 历史记录：每轮规则表现
        self.current_round = 0  # 轮次计数器
        self.weight_update_period = 20  # 权重更新周期（每5轮更新一次）
        self.smoothing_factor = 0.05  # 权重更新平滑因子（避免突变）

        logger.info(f"初始规则权重：{self.rule_weights}")
        logger.info(f"权重更新周期：每{self.weight_update_period}轮")

        dataset = load_from_disk(args.data_path)
        train_list = list(dataset)
        for line in train_list:
            self.prompt2answer[line['context'].strip()] = line['answer'].strip().split('\n')
            # print('context:', line['context'].strip())
            # print('answer:', self.prompt2answer[line['context'].strip()])

    def set_rule_weights(self, new_weights: List[float]):
        """更新规则权重"""
        if len(new_weights) != self.rule_num:
            raise ValueError(f"需要{self.rule_num}个权重，实际提供{len(new_weights)}个")
        # 权重归一化
        weight_sum = sum(new_weights)
        if weight_sum <= 0:
            raise ValueError("权重总和必须为正数")
        self.rule_weights = [w / weight_sum for w in new_weights]
        logger.info(f"更新规则权重：{self.rule_weights}")


    def update_weights_based_on_history(self):
        """基于历史表现自动更新权重"""
        if len(self.hist) < self.weight_update_period:
            return  # 历史数据不足，不更新

        # 提取最近N轮的历史数据
        recent_hist = self.hist[-self.weight_update_period:]

        # 计算各规则总使用率和成功率
        total_usage = [0] * self.rule_num
        total_success = [0] * self.rule_num
        for record in recent_hist:
            tats = record["rule_stats"]
            for i in range(self.rule_num):
                total_usage[i] += tats["usage"][i]
                total_success[i] += tats["succ"][i]

        # 计算成功率（避免除零）
        success_rate = []
        for i in range(self.rule_num):
            if total_usage[i] == 0:
                success_rate.append(0)  # 未使用过的规则成功率设为0
            else:
                success_rate.append(total_success[i] / total_usage[i])
        # succ_sum = sum(success_rate)
        # if succ_sum > 0:
        #     success_rate = [s / succ_sum for s in success_rate]


        # 计算使用率占比（归一化）
        total_usage_sum = sum(total_usage)
        usage_ratio = []
        for i in range(self.rule_num):
            if total_usage_sum == 0:
                usage_ratio.append(1.0 / self.rule_num)  # 平均分配
            else:
                usage_ratio.append(total_usage[i] / total_usage_sum)

        # 综合成功率和使用率计算新权重（成功率权重占比alpha，使用率1-alpha）
        alpha = 0.5
        combined_score = [alpha * sr + (1-alpha) * ur for sr, ur in zip(success_rate, usage_ratio)]
        score_sum = sum(combined_score)
        if score_sum <= 0:
            logger.warning("综合得分总和为零，使用初始权重")
            new_weights = self.initial_weights.copy()
        else:
            new_weights = [s / score_sum for s in combined_score]
            # new_weights = [s  for s in combined_score]


        # 平滑更新（结合当前权重避免突变）
        self.rule_weights = [
            self.smoothing_factor * new_w + (1 - self.smoothing_factor) * old_w
            for new_w, old_w in zip(new_weights, self.rule_weights)
        ]

        logger.info(f"自动更新规则权重：{self.rule_weights}")
        logger.info(f"最近{self.weight_update_period}轮成功率：{success_rate}")
        logger.info(f"最近{self.weight_update_period}轮使用率：{usage_ratio}")
        
        # 记录权重更新日志
        logger.info(f"权重更新: 基于最近{self.weight_update_period}轮")
        logger.info(f"规则使用率: {total_usage}")
        logger.info(f"使用率占比: {usage_ratio}")
        logger.info(f"规则成功率: {success_rate}")
        logger.info(f"综合得分: {combined_score}")
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

    def rule_check_old(self,action_chain,init_state=None,init_clear=None,goal_state=None):
        current_support = init_state.copy()
        current_clear = init_clear.copy()
        
        hit_condition = [0] * self.rule_num
        use_rule = [0] * self.rule_num
        hand_state = 0
        for action in action_chain:
            if hand_state == 1:
                hit_condition[0] += 1
                if 'unstack' in action or 'pick-up' in action:
                    use_rule[0]+= 1
            if hand_state == 0:
                hit_condition[1] += 1
                if ('stack' in action and 'unstack' not in action) or 'put-down' in action:
                    use_rule[1] += 1

            if 'unstack' in action or 'pick-up' in action:
                hand_state = 1   
            
            if ('stack' in action and 'unstack' not in action) or 'put-down' in action:
                hand_state = 0
        return use_rule, hit_condition

    def rule_check(self, action_chain, init_state, goal_state):
        # 初始化状态
        current_support = init_state['support'].copy()  # 块->支撑物
        current_clear = init_state['clear'].copy()      # 当前clear的块集合
        hand_state = 0  # 0:空, 1:持有
        holding = None  # 当前持有的块

        # 规则定义（都是负向规则）：
        # 0: hand_state==1时才能pick up/unstack
        # 1: hand_state==0时才能stack/putdown
        # 2: 执行stack动作前，目标块必须clear 暂无
        # 3: 执行unstack动作前，被拿的块必须clear 暂无
        # 4: pick up动作只能对在桌面且clear的块执行  暂无

        use_rule = [0] * self.rule_num
        succ_rule = [0] * self.rule_num
        # 规则统计：[触发次数，成功次数]

        for action in action_chain:
            action = action.strip().lower()

            try:
                if '(pick-up ' in action:
                    blk = re.search(r'\(pick-up (\w+)\)', action).group(1)
                    action_type = 'pick-up'
                elif '(put-down ' in action:
                    blk = re.search(r'\(put-down (\w+)\)', action).group(1)
                    action_type = 'put-down'
                elif '(unstack ' in action:
                    match = re.search(r'\(unstack (\w+) (\w+)\)', action)
                    blk1, blk2 = match.group(1), match.group(2)
                    action_type = 'unstack'
                elif '(stack ' in action:
                    match = re.search(r'\(stack (\w+) (\w+)\)', action)
                    blk1, blk2 = match.group(1), match.group(2)
                    action_type = 'stack'
                else:
                    continue

                # 规则检查
                if hand_state == 0:
                    use_rule[0] += 1
                    if action_type in ['pick-up', 'unstack']:
                        succ_rule[0] += 1

                if hand_state == 1:
                    use_rule[1] += 1
                    if action_type in ['stack', 'put-down']:
                        succ_rule[1] += 1


                # if action_type in ['pick-up', 'unstack']:
                #     use_rule[0] += 1
                #     if hand_state == 0:
                #         succ_rule[0] += 1

                # if action_type in ['stack', 'put-down']:
                #     use_rule[1] += 1
                #     if hand_state == 1:
                #         succ_rule[1] += 1

                # if action_type == 'stack':
                #     use_rule[2] += 1
                #     if blk2 in current_clear:
                #         succ_rule[2] += 1

                # if action_type == 'unstack':
                #     use_rule[3] += 1
                #     if blk1 in current_clear:
                #         succ_rule[3] += 1

                # if action_type == 'pick-up':
                #     use_rule[4] += 1
                #     on_table = current_support.get(blk) == 'table'
                #     is_clear = blk in current_clear
                #     if on_table and is_clear:
                #         succ_rule[4] += 1
                
               # 更新状态
                if action_type == 'pick-up' and hand_state == 0:
                    if blk in current_clear:
                        current_clear.remove(blk)
                    holding = blk
                    hand_state = 1
                    current_support.pop(blk, None)

                elif action_type == 'put-down' and hand_state == 1 and holding == blk:
                    current_support[blk] = 'table'
                    current_clear.add(blk)
                    holding = None
                    hand_state = 0

                elif action_type == 'stack' and hand_state == 1 and holding == blk1:
                    current_support[blk1] = blk2
                    if blk2 in current_clear:
                        current_clear.remove(blk2)
                    current_clear.add(blk1)
                    holding = None
                    hand_state = 0

                elif action_type == 'unstack' and hand_state == 0:
                    current_support.pop(blk1, None)
                    # 检查支撑物是否变为clear
                    has_other = any(v == blk2 for v in current_support.values())
                    if not has_other:
                        current_clear.add(blk2)
                    if blk1 in current_clear:
                        current_clear.remove(blk1)
                    holding = blk1
                    hand_state = 1




            except Exception as e:
                logger.warning(f"动作解析错误: {action}, 错误: {e}")
                continue



        return use_rule, succ_rule


    
    def init_reward_info(self,init_state, goal_state):
        """
        在训练开始时调用，预计算哪些块在初始状态就与目标不符（规则 [1]）。
        """
        wrong_blocks = {
            block
            for block, support in init_state.items()
            if goal_state.get(block) != support
        }
        # 记录每个块的目标支持方，便于分层检查
        target_support = goal_state.copy()
        return {
            "wrong_blocks": wrong_blocks,
            "target_support": target_support
        }

    def extract_states(self, text: str):
        # 分割初始与目标部分
        parts = re.split(r'\.\s*My goal is to have that', text, maxsplit=1)
        init_text = parts[0] if len(parts) > 0 else ""
        goal_text = parts[1] if len(parts) > 1 else ""

        # 匹配模式
        clear_pattern = re.compile(r'the (\w+) block is clear', re.IGNORECASE)
        ontop_pattern = re.compile(r'the (\w+) block is on top of the (\w+) block', re.IGNORECASE)
        table_pattern = re.compile(r'the (\w+) block is on the table', re.IGNORECASE)

        # 解析初始状态
        init_support: Dict[str, str] = {}
        init_clear: Set[str] = set()

        for blk in clear_pattern.findall(init_text):
            init_clear.add(blk.lower())

        for blk, sup in ontop_pattern.findall(init_text):
            blk = blk.lower()
            sup = sup.lower()
            init_support[blk] = sup
            if sup in init_clear:
                init_clear.remove(sup)

        for blk in table_pattern.findall(init_text):
            blk = blk.lower()
            init_support[blk] = 'table'

        # 确保支撑物状态正确
        for blk in init_support:
            sup = init_support[blk]
            if sup != 'table' and sup in init_support and sup in init_clear:
                init_clear.remove(sup)

        # 解析目标状态
        goal_support: Dict[str, str] = {}
        for blk, sup in ontop_pattern.findall(goal_text):
            blk = blk.lower()
            sup = sup.lower()
            goal_support[blk] = sup

        for blk in table_pattern.findall(goal_text):
            blk = blk.lower()
            goal_support[blk] = 'table'

        return {'support': init_support, 'clear': init_clear}, goal_support



    #todo: rewrite correctness_score
    def correctness_score(self, prompt, response):
        init_state,  goal_state = self.extract_states(prompt)
        current_rule = RuleTaj(self.rule_num, self.rule_weights)


        try:
            # pred = response.split('[PLAN]')[1].split('[PLAN END]')[0]
            pred = response.split('[PLAN END]')[0]
        except Exception as e:
            return 0.0,0.0



        pred = parse_pred(pred)
        if not pred:
            return 0.0, 0.0

        # 检查规则遵守情况
        use_rule, succ_rule = self.rule_check(action_chain=pred, init_state=init_state, goal_state=goal_state)
        
        current_rule.usage = use_rule
        current_rule.succ = succ_rule

        # 计算规则成功率
        rule_success = []
        for u, s in zip(use_rule, succ_rule):
            if u == 0:
                rule_success.append(1.0)  # 未适用规则视为成功
            else:
                rule_success.append(s / u)


        rule_score =   sum(r * w for r, w in zip(rule_success, self.rule_weights))
         
        
        reward = 1.0
        truth = self.prompt2answer[prompt]
        if len(truth)!= len(pred):
            reward = 0.0
            # return rule_score,0.0

        for i in range(0,len(truth)):
            try:
                if truth[i] != pred[i]:
                    # print('truth:',truth[i])
                    # print('pred:',pred[i])
                    error_lines += 1
                    reward = 0.0
            except Exception as e:
                reward = 0.0


        self.hist.append({
            "rule_stats": current_rule.to_dict(),
            "outcome_reward": reward
        })
        print(self.current_round)
        print(current_rule.to_dict())
        
        return rule_score,reward








    def evaluate_result(self, query):
        response = ''
        try:
            with timeout(self.timeout_seconds):
                if args.template_type == "qwen":
                    prompt = \
                    query.split("<|im_end|>\n<|im_start|>user\n")[-1].split("<|im_end|>\n<|im_start|>assistant\n")[
                        0].strip()
                    # print('query:',query)
                    # prompt = query.split("<|eot_id|><|start_header_id|>user<|end_header_id|>")[-1].split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[0].strip()
                    # distill = True
                    # if distill:
                    #     prompt = query.split("")
                    # prompt = query.split("<｜begin▁of▁sentence｜>")
                    # print('prompt',prompt)
                    response = query.split("<|im_end|>\n<|im_start|>assistant\n")[-1]
                    # print('response',response)
                    if "<|im_end|>" not in response and "<|endoftext|>" not in response:
                        return -1.0
                    response = query.split("<|im_end|>\n<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].split(
                        "<|endoftext|>")[0].strip()
                elif args.template_type == 'llama':
                    prompt =  query.split("<|eot_id|><|start_header_id|>user<|end_header_id|>")[-1].split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[0].strip()
                    # print("prompt:",prompt)
                    response = query.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[-1].strip()
                    # print('response:',response)
                    # print()
                    if "<|eot_id|>" not in response:
                        return -1.0
                    response = query.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
                    # print("response:",response)

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
        rule_score,reward = self.correctness_score(prompt, output)

        self.current_round += 1
        logger.info(f"current_round数： {self.current_round}")
        # 权重调整
        if self.current_round % self.weight_update_period == 0:
            self.update_weights_based_on_history()


        print('rule_score:',rule_score)
        print('reward:',reward)

        return 0.2*rule_score+0.8*reward
        # return reward

        # return 0.4*rule_score+1.6*reward

    
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
    parser.add_argument("--data_path", type=str, default='/data/coding/train_blocks')    # for
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--max_len", type=int, default="2048")

    parser.add_argument("--port", type=int, default=5663, help="Port number for the server")
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
