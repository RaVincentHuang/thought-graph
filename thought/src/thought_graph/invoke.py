import re
import os
from openai import OpenAI
from typing import Tuple, Union, List, Iterator, Optional

from thought_graph.utils import get_logger

from thought_graph.trace import global_trace

import logging

logger = get_logger(__name__)

class QueryContext:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE_URL"))
        self.model = os.getenv("OPENAI_API_MODEL", "gpt-4o")
        self.prompt_buffer: list[str] = []
        self._depth: int = 0 

    @property
    def is_active(self) -> bool:
        return self._depth > 0

    def enter(self):
        if self._depth == 0:
            self.prompt_buffer.clear()
        self._depth += 1

    def exit(self):
        self._depth -= 1
        if self._depth == 0:
            self.prompt_buffer.clear()

_query_ctx = QueryContext()

def _build_regex_from_template(formatted_template: str) -> str:
    """
    将填充后的模板转换为宽容的正则表达式。
    例如: "We have 10=[n1]+[n2]" 
    -> r"We\ have\ 10\=\s*<(.*?)>\s*\+\s*<(.*?)>"
    """
    # 1. 转义特殊字符
    pattern = re.escape(formatted_template)
    
    # 2. 替换占位符 \[.*?\] 为捕获组 <(.*?)>
    #    允许捕获组周围有空格
    pattern = re.sub(r'\\\[.*?\\\]', r'\\s*<(.*?)>\\s*', pattern)
    
    # 3. [关键] 允许行首有列表符号 (如 "1. ", "- ") 或缩进
    #    这样 LLM 如果输出了 numbered list 也能被解析
    return r'^\s*(?:[\d\-\*\.]+\s*)?' + pattern

def _run_llm(context_text: str, instruction: str, temperature: float = 0.0) -> str:
    """
    底层 LLM 通信函数
    """
    logger.debug(f"LLM Context:\n{context_text}\n---\nInstruction:\n{instruction}")
    response = _query_ctx.client.chat.completions.create(
        model=_query_ctx.model,
        messages=[
            # System Prompt: 确立角色和输出协议
            {"role": "system", "content": (
                "You are a precise reasoning engine."
                "You must strictly follow the output format requirements. "
                "Always wrap your final result between <OUTPUT> and </OUTPUT> tags."
            )},
            {"role": "user", "content": context_text + "\n\n" + instruction}
        ],
        temperature=temperature
    )
    
    llm_output = response.choices[0].message.content
    assert llm_output is not None
    logger.debug(f"LLM Response:\n{llm_output}")
    return llm_output

def _extract_content_block(llm_output: str) -> str:
    """提取 <OUTPUT>...</OUTPUT> 之间的内容"""
    if "<OUTPUT>" in llm_output:
        # 支持闭合标签，提高精确度
        content = llm_output.split("<OUTPUT>")[-1]
        if "</OUTPUT>" in content:
            content = content.split("</OUTPUT>")[0]
        return content.strip()
    return llm_output

def query(prompt_template: str, *args) -> Union[str, Tuple[str, ...], None]:
    """
    [Single Mode] 强迫 LLM 收敛，只输出唯一且确定的结果。
    """
    formatted_text = prompt_template.format(*args)
    capture_targets = re.findall(r'\[(.*?)\]', prompt_template)
    
    if not capture_targets:
        _query_ctx.prompt_buffer.append(formatted_text)
        return None
    
    # 构造单次输出的样本
    output_sample = prompt_template.replace('[', '<').replace(']', '>').format(*args)
    
    # [Prompt Engineering] 针对单次输出的强化指令
    instruction = (
        "--- INSTRUCTIONS ---\n"
        "1. Complete the thought based on the context.\n"
        "2. Provide **EXACTLY ONE** single result.\n"
        "3. **SYNTAX ENFORCEMENT (CRITICAL)**:\n"
        "   - You MUST surround your filled values with angle brackets `<` and `>`.\n"
        "   - The brackets are PART of the answer format. DO NOT remove them.\n"
        "   - Example: If the value is '5', you must write `<5>`, NOT `5`.\n"
        "4. Your output must strictly match this template:\n"
        f"   {output_sample}\n"
        "5. **DO NOT** output a list. **DO NOT** provide multiple options.\n"
        "6. Final Answer Format:\n"
        f"<OUTPUT>\n{output_sample}\n</OUTPUT>"
    )
    
    full_context = "\n".join(_query_ctx.prompt_buffer + [formatted_text])
    
    # 温度设为 0，追求确定性
    llm_output = _run_llm(full_context, instruction, temperature=0.0)
    result_part = _extract_content_block(llm_output)

    # 解析
    # 这里的正则不需要太复杂，因为只提取一次
    # 为了兼容 _build_regex 的逻辑，我们也可以复用它，但简单的 findall 也可以
    captured_values = re.findall(r'\<(.*?)\>', result_part)
    
    # 清理 Buffer
    _query_ctx.prompt_buffer.clear()
    
    if len(captured_values) == 0: return None
    if len(captured_values) == 1: return captured_values[0]
    return tuple(captured_values)

def query_iter(prompt_template: str, *args) -> Iterator[Union[str, Tuple[str, ...]]]:
    """
    [Iterative Mode] 强迫 LLM 发散，输出列表，并解析为迭代器。
    """
    formatted_text = prompt_template.format(*args)
    capture_targets = re.findall(r'\[(.*?)\]', prompt_template)
    
    if not capture_targets:
        _query_ctx.prompt_buffer.append(formatted_text)
        return iter([])

    output_sample = prompt_template.replace('[', '<').replace(']', '>').format(*args)
    
    
    # 1. 立即消费之前积累的 Context Buffer (例如之前的 query 调用)
    buffered_deps = global_trace.consume_query_dependencies()
    
    # 2. 将这些依赖“冻结”在 global_trace 的暂存区中
    # 这样，instrument.py 在处理后续的每一次循环时，都能读到这份数据
    global_trace.set_active_iter_deps(buffered_deps)
    
    # [Prompt Engineering] 针对多次输出的强化指令
    instruction = (
        "--- INSTRUCTIONS ---\n"
        "1. Brainstorm and list **MULTIPLE DISTINCT** possibilities/results based on the context.\n"
        "2. Output each result on a **NEW LINE**.\n"
        "3. **SYNTAX ENFORCEMENT (CRITICAL)**:\n"
        "   - You MUST surround your filled values with angle brackets `<` and `>`.\n"
        "   - The brackets are PART of the answer format. DO NOT remove them.\n"
        "   - Example: If the value is '5', you must write `<5>`, NOT `5`.\n"
        "4. Each line must strictly match this template:\n"
        f"   {output_sample}\n"
        "5. Final Answer Format (Example):\n"
        "<OUTPUT>\n"
        f"{output_sample}\n"
        f"{output_sample}\n"
        "...\n"
        "</OUTPUT>"
    )
    
    full_context = "\n".join(_query_ctx.prompt_buffer + [formatted_text])
    
    # 温度设为 0.7，鼓励发散性
    llm_output = _run_llm(full_context, instruction, temperature=0.7)
    result_part = _extract_content_block(llm_output)

    # 构造行级匹配正则
    # 增加 re.MULTILINE 标志，确保 ^ 能匹配每一行的开头
    regex_pattern = _build_regex_from_template(formatted_text)
    
    # 使用 finditer 或 findall
    # 由于正则中包含了 ^，我们需要针对 result_part 按行处理，或者使用 re.MULTILINE
    matches = re.findall(regex_pattern, result_part, flags=re.MULTILINE)
    
    logger.debug(f"Iterative matches found: {len(matches)}")
    
    _query_ctx.prompt_buffer.clear()
    
    for match in matches:
        if not match: continue
        yield match