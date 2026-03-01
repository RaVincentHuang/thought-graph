from thought_graph import query, analysis, query_iter
from thought_graph.trace import ProgramTrace, TraceEvent
from thought_graph.graph import DataFlowGraph, ThoughtGraph
from thought_graph.analysis import AnalysisOutput

@analysis(output_type=AnalysisOutput.THOUGHT)
def example1(num):
    query("Input number {}", num)
    num1, num2 = query("We have {}=[num1]+[num2]", num) # pyright: ignore[reportGeneralTypeIssues]
    num_prime = num1 + num2
    print(f"Decomposed {num} into {num1} and {num2}, sum is {num_prime}")
    return num1, num2

@analysis
def update(nums: list[int], n1: int, n2: int, res: int) -> list[int]:
    """
    纯函数：通过一次枚举逻辑生成新的状态。
    输入：nums=[1,3,6,7], n1=7, n2=3, res=4
    输出：[6, 1, 4]
    """
    # 确保是纯函数：不改变原 nums，一次性构造新列表
    # 逻辑：保留不在本次操作中的数字，并加上结果
    found_n1, found_n2 = False, False
    new_nums = []
    
    for x in nums:
        if x == n1 and not found_n1:
            found_n1 = True
            continue
        if x == n2 and not found_n2:
            found_n2 = True
            continue
        new_nums.append(x)
        
    new_nums.append(res)
    return new_nums

@analysis
def analysis_status(nums: list[int]) -> float:
    score_str = query("Evaluate the potential of these numbers {} to reach 24. ""Provide a confidence score from 0 to 10: [score]", nums)
    # 记录该 DEF 事件，标记为 LLM 定义的评分节点 [cite: 231, 399]
    return float(score_str)

@analysis(output_type=AnalysisOutput.THOUGHT)
def solve_game24_bfs(initial_numbers: list[int]):
    """
    使用 BFS 逻辑实现 Game of 24 的 ToT 推理
    """
    # 初始状态：当前的数字集合
    states = [initial_numbers]
    
    # Game of 24 在 BFS 逻辑下通常进行 3 轮迭代 [cite: 597, 598]
    for step in range(3):
        print(f"Step {step + 1}")
        next_states = []
        
        for nums in states:
            # 1. 积累背景：说明任务目标 [cite: 231, 308]
            query("Goal: Use numbers to get 24. Current numbers: {}", nums)
            
            # 2. 触发独立 Invoke：捕获两个操作数和结果 [cite: 231, 256]
            # 假设 LLM 严格遵守 <OUTPUT> 协议返回 [n1][n2][res]
            n1, n2, res = query("Pick two numbers from {} and their sum/sub/mul/div result: [n1], [n2], [res]", nums)
            
            # 3. 更新状态：移除已使用的数字，加入新结果 [cite: 256, 347]
            new_nums = update(nums, int(n1), int(n2), int(res))
            
            next_states.append(new_nums)
            print(f"Action: {n1} & {n2} -> {res} | New State: {new_nums}")

        candidates = sorted(next_states, key=lambda ns: analysis_status(ns), reverse=True)
        # BFS 宽度限制：每层保留前 3 个状态 [cite: 598]
        states = candidates[:3]

    # 最终验证 [cite: 620]
    success = any(24 in s for s in states)
    print(f"Final Success: {success}")
    return success

@analysis
def analysis_nums(nums: str) -> float:
    score_str = query("Evaluate the potential of these numbers {} to reach 24. ""Provide a confidence score from 0 to 10: [score]", nums)
    return float(score_str)

# nums: '3, 8, 3, 3'
@analysis(output_type=AnalysisOutput.THOUGHT)
def game24(nums: str) -> bool:
    states: list[str] = [nums]
    for step in range(3):
        next_states: list[str] = []
        for state in states:
            query("Goal: Use numbers to get 24. Current numbers: {}", state)
            query("Pick two numbers from {} and their sum/sub/mul/div", state)
            for child in query_iter("result in new numbers: [numbers]"):
                next_states.append(child)
        candidates: list[str] = sorted(next_states, key=lambda ns: analysis_nums(ns), reverse=True)
        states = candidates[:3]
    success = any("24" in s.split(", ") for s in states)
    print(f"Final Success: {success}")

    return success

def print_trace_graph(trace: ProgramTrace, title: str):
    """
    辅助函数：可视化打印 Trace 数据流图
    """
    print(f"\n{'='*20} {title} Trace Visualization {'='*20}")
    print(f"Total Events: {len(trace.events)}")
    
    for i, event in enumerate(trace.events):
        # 格式化时间戳
        time_offset = f"{event.timestamp - trace.events[0].timestamp:.4f}s"
        
        # 格式化 USE 节点列表
        uses_str = ", ".join([str(u) for u in event.use_nodes]) if event.use_nodes else "None"
        
        # 格式化 DEF 节点
        def_str = str(event.def_node)
        
        # 打印类似于汇编的指令流: DEF <- USE [Type]
        print(f"[{i:03d}][{time_offset}] {event.stmt_type.upper().ljust(12)} : {def_str} <--- [{uses_str}]")

if __name__ == "__main__":
    # print("Running Example 1...")
    
    # # [关键] 最外层调用会自动拆包：返回 (结果, Trace对象)
    # # 这里的 100 是输入参数
    # result_ex1, thought_ex1 = example1(100) 

    # thought_ex1.print_summary()
    # # ------------------------------------------------------------------
    # # 场景 2: 调用复杂任务 Game of 24
    # # ------------------------------------------------------------------
    # print("\n\nRunning Game of 24 BFS...")
    
    # input_nums = [4, 1, 8, 7] # 一个有解的例子
    
    # # 同样，最外层调用返回 (bool, Trace对象)
    # is_success, thought_game24 = solve_game24_bfs(input_nums)
    
    # thought_game24.print_summary()
    
    # print(f"Game Solved: {is_success}")
    
    is_success, thought_game24 = game24("3, 8, 3, 3")
    thought_game24.print_summary()
    
    print(f"Game Solved (str nums): {is_success}")
    
    # 打印庞大的数据流图
    # 你会看到从 input_nums 到 n1, n2, res, 再到 update 生成的 new_nums 的完整链路
    