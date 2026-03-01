from thought_graph import query, analysis, query_iter

@analysis
def example1(num):
    query("Input number {}", num)
    num1, num2 = query("We have {}=[num1]+[num2]", num) # pyright: ignore[reportGeneralTypeIssues]
    num_prime = num1 + num2
    print(f"Decomposed {num} into {num1} and {num2}, sum is {num_prime}")

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

@analysis
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
@analysis
def game24(nums: str) -> bool:
    states: list[str] = [nums]
    for step in range(3):
        next_states: list[str] = []
        for state in states:
            query("Goal: Use numbers to get 24. Current numbers: {}", state)
            query("Pick two numbers from {} and their sum/sub/mul/div", state)
            for nums in query_iter("result in new numbers: [numbers]"):
                next_states.append(nums)
        candidates: list[str] = sorted(next_states, key=lambda ns: analysis_nums(ns), reverse=True)
        states = candidates[:3]
    success = any("24" in s.split(", ") for s in states)
    print(f"Final Success: {success}")

    return success

if __name__ == "__main__":
    example1(10)
    solve_game24_bfs([3, 8, 3, 3])
    game24("3, 8, 3, 3")