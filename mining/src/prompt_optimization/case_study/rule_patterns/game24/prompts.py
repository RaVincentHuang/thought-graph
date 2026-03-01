zero_shot = """
【Task】
In the Game 24, you are given numbers. Your goal is to find arithmetic expressions that evaluate to 24 using all numbers exactly once, with only basic arithmetic operations (+, -, *, /) between two numbers at a time.

【Rules】
1. Use each given number exactly once
2. Only use basic arithmetic operations: +, -, *, /
3. Operations can only be performed between two numbers at a time
4. You can use parentheses to control the order of operations
5. Division must result in a whole number
6. The answer must be in the format of "Answer: (expression) = 24" without any other explanation.

【Examples】
Input: 4 4 6 8
Answer: (6 - 4) * (4 + 8) = 24

Input: 2 9 10 12
Answer: (12 * 2) * (10 - 9) = 24

Input: 4 9 10 13
Answer: 4 * (9 - (13 - 10)) = 24

Input: 1 4 8 8
Answer: (1 + 8 / 4) * 8 = 24

Input: 5 5 5 9
Answer: ((5 + 5) + 5) + 9 = 24

Input: {QUESTION}
"""

few_shots = """
【Task】
In the Game 24, you are given numbers. Your goal is to find arithmetic expressions that evaluate to 24 using all numbers exactly once, with only basic arithmetic operations (+, -, *, /) between two numbers at a time.

【Rules】
1. Use each given number exactly once
2. Only use basic arithmetic operations: +, -, *, /
3. Operations can only be performed between two numbers at a time
4. You can use parentheses to control the order of operations
5. Division must result in a whole number
6. You should think step by step and follow the steps to get the answer.
7. The answer must be in the format of "Answer: (expression) = 24" without any other explanation.

【Examples】
Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24

Input: 2 9 10 12
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24

Input: 4 9 10 13
Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24

Input: 1 4 8 8
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24

Input: 5 5 5 9
Steps:
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24

Input: {QUESTION}
"""
# version1
ours_v1 = """
【Task】
In the Game 24, you are given numbers. Your goal is to find arithmetic expressions that evaluate to 24 using all numbers exactly once, with only basic arithmetic operations (+, -, *, /) between two numbers at a time.

【Rules】
1. Use each given number exactly once
2. Only use basic arithmetic operations: +, -, *, /
3. Operations can only be performed between two numbers at a time
4. You can use parentheses to control the order of operations
5. Division must result in a whole number
6. The answer must be in the format of "Answer: (expression) = 24" without any other explanation.
7. You should think step by step and follow the steps to get the answer. Additionally, throughout your thinking process, these guidelines can help you:
[1] At the outset, carefully observe and prioritize how to form the factors of 24, that is, in the first two steps, focus on how to create combinations that yield the factors of 24, such as 1 and 24, 2 and 12, 3 and 8, 4 and 6, and then proceed to multiplication directly in the third step.
[2] When it is difficult to form the factors of 24, consider the possibility of using addition or subtraction in the final step to combine the numbers.

【Examples】
Input: 4 4 6 8
Steps: 
According to guideline[1]，we find that we can form the combination of 2 * 12.
8 + 4 = 12 （left: 4 6 12）
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24

Input: 2 9 10 12
Steps:
According to guideline[1]，we find that we can form the combination of 1 * 24.
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24

Input: 4 9 10 13
Steps:
According to guideline[1]，we find that we can form the combination of 4 * 6.
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24

Input: 1 4 8 8
Steps:
According to guideline[1]，we find that we can form the combination of 3 * 8.
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24

Input: 5 5 5 9
Steps:
According to guideline[2]，we notice that the distribution of the original numbers deviates from the factors of 24, so we might consider more the possibility of using addition in the final step.
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24

Input: {QUESTION}"""


# version2
ours_v2 = """
【Task】
In the Game 24, you are given numbers. Your goal is to find arithmetic expressions that evaluate to 24 using all numbers exactly once, with only basic arithmetic operations (+, -, *, /) between two numbers at a time.

【Rules】
1. Use each given number exactly once
2. Only use basic arithmetic operations: +, -, *, /
3. Operations can only be performed between two numbers at a time
4. You can use parentheses to control the order of operations
5. Division must result in a whole number
6. You should think step by step and follow the steps to get the answer.
7. The answer must be in the format of "Answer: (expression) = 24" without any other explanation.

【Guidelines】
[1] Prioritize forming factors of 24 (e.g., 3×8, 4×6) in the first two steps
[2] If factors are hard to form, consider addition/subtraction in the final step
[3] Avoid decimal intermediates; prefer integer arithmetic unless division is strictly necessary and results in a whole number
[4] Backtrack proactively: If an operation violates rules (e.g., decimal result, unused numbers), revert and try other combinations. Moreover, at every step, explicitly check if any extra numbers are introduced or existing numbers are omitted. If yes, backtrack immediately.

【Examples】
Input: 3 5 7 8
Steps:
1. Attempt 1:
   5 + 7 = 12 (left: 3, 8, 12)
   Check: No extra numbers. Next, try 3 * 8 = 24 (left: 12, 24)
   Invalid: 24 - 12 = 12 not equal to 24 Backtrack!

2. Attempt 2:
   8 - 5 = 3 (left: 3, 3, 7)
   Check: No extra numbers. Next, 3 * 7 = 21 (left: 3, 21)
   3 + 21 = 24 (left: 24). Valid! According to guidance[2]
Answer: 3 + (8 - 5) * 7 = 24

Input: 2 3 4 9
Steps:
1. Attempt 1:
   9 - 2 = 7 (left: 3, 4, 7)
   7 * 3 = 21 (left: 4, 21)
   21 + 4 = 25 not equal to 24  Backtrack!

2. Attempt 2:
   9 / 3 = 3 (left: 2, 4, 3)
   4 * 3 = 12 (left: 2, 12)
   12 * 2 = 24 (left: 24) According to guidance[1]
Answer: (9 / 3) * 4 * 2 = 24

Input: {QUESTION}
"""

# ours_v3
ours_v3 = """
【Task】
In the Game 24, you are given numbers. Your goal is to find arithmetic expressions that evaluate to 24 using all numbers exactly once, with only basic arithmetic operations (+, -, *, /) between two numbers at a time.

【Rules】
1. Use each given number exactly once
2. Only use basic arithmetic operations: +, -, *, /
3. Operations can only be performed between two numbers at a time
4. You can use parentheses to control the order of operations
5. Division must result in a whole number
6. You should think step by step and follow the steps to get the answer.
7. The answer must be in the format of "Answer: (expression) = 24" without any other explanation.

【Guidelines】
[1] At the outset, carefully observe and prioritize how to form the factors of 24, that is, in the first two steps, focus on how to create combinations that yield the factors of 24, such as 1 and 24, 2 and 12, 3 and 8, 4 and 6, and then proceed to multiplication directly in the third step.  
[2] When it is difficult to form the factors of 24, consider the possibility of using addition or subtraction in the final step to combine the numbers.  
[3] Avoid decimal intermediates; prefer integer arithmetic unless division is strictly necessary and results in a whole number.
[4] Backtrack proactively: If an operation violates rules (e.g., decimal result, unused numbers), revert and try other combinations. Moreover, at every step, explicitly check if any extra numbers are introduced or existing numbers are omitted. If yes, backtrack immediately.

【Examples】
Input: 4 4 6 8
Steps: 
According to guideline[1]，we find that we can form the combination of 2 * 12.
8 + 4 = 12 （left: 4 6 12）
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24

Input: 1 4 8 8
Steps:
According to guideline[1]，we find that we can form the combination of 3 * 8.
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24

Input: 5 5 5 9
Steps:
According to guideline[2]，we notice that the distribution of the original numbers deviates from the factors of 24, so we might consider more the possibility of using addition in the final step.
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24

Input: 3 5 7 8
Steps:
1. Attempt 1:
   5 + 7 = 12 (left: 3, 8, 12)
   Check: No extra numbers. Next, try 3 * 8 = 24 (left: 12, 24)
   Invalid: 24 - 12 = 12 not equal to 24 Backtrack!

2. Attempt 2:
   8 - 5 = 3 (left: 3, 3, 7)
   Check: No extra numbers. Next, 3 * 7 = 21 (left: 3, 21)
   3 + 21 = 24 (left: 24). Valid! According to guidance[2]
Answer: 3 + (8 - 5) * 7 = 24

Input: 2 3 4 9
Steps:
1. Attempt 1:
   9 - 2 = 7 (left: 3, 4, 7)
   7 * 3 = 21 (left: 4, 21)
   21 + 4 = 25 not equal to 24  Backtrack!

2. Attempt 2:
   9 / 3 = 3 (left: 2, 4, 3)
   4 * 3 = 12 (left: 2, 12)
   12 * 2 = 24 (left: 24) According to guidance[1]
Answer: (9 / 3) * 4 * 2 = 24

Input: {QUESTION}
"""

# version4
ours_v4 = """
【Task】
In the Game 24, you are given numbers. Your goal is to find arithmetic expressions that evaluate to 24 using all numbers exactly once, with only basic arithmetic operations (+, -, *, /) between two numbers at a time.

【Rules】
1. Use each given number exactly once
2. Only use basic arithmetic operations: +, -, *, /
3. Operations can only be performed between two numbers at a time
4. You can use parentheses to control the order of operations
5. Division must result in a whole number
6. You should think step by step and follow the steps to get the answer.
7. The answer must be in the format of "Answer: (expression) = 24" without any other explanation.

【Guidelines】
[1] At the outset, carefully observe and prioritize how to form the factors of 24, that is, in the first two steps, focus on how to create combinations that yield the factors of 24, such as 1 and 24, 2 and 12, 3 and 8, 4 and 6, and then proceed to multiplication directly in the third step.  
[2] When it is difficult to form the factors of 24, consider the possibility of using addition or subtraction in the final step to combine the numbers.  

【Good thinking patterns】
(1)
1 * 24 = 24 (left: 24)
(2)
2 * 12 = 24 (left: 24)
(3)
3 * 8 = 24 (left: 24)
(4)
4 * 6 = 24 (left: 24)
(5)
12 - 4 = 8 (left: 3 8)
3 * 8 = 24 (left: 24)
(6)
10 * 12 = 120 (left: 5 120)
120 / 5 = 24 (left: 24)
(7)
2 + 4 = 6 (left: 6 12 6)
6 + 12 = 18 (left: 6 18)
6 + 18 = 24 (left: 24)

【Bad thinking patterns】
(1)
6 / 52 = 0.1154 (left: 0.1154)
(2)
10 - 8 = 2 (left: 5 5 2)  
5 - 2 = 3 (left: 5 3)  
5 * 3 = 24 (left: 24) 
(3)
2 + 7 = 9 (left: 2 12 9)
2 + 9 = 11 (left: 12 11)
12 + 9 = 21 (left: 2 21)

【Good Examples】
Input: 3 5 7 8
Steps:
1. Attempt 1:
   5 + 7 = 12 (left: 3, 8, 12)
   Check: No extra numbers. Next, try 3 * 8 = 24 (left: 12, 24)
   Invalid: 24 - 12 = 24 (left: 24), Calculation error! not equal to 24 Backtrack!

2. Attempt 2:
   8 - 5 = 3 (left: 3, 3, 7)
   Check: No extra numbers. Next, 3 * 7 = 21 (left: 3, 21)
   3 + 21 = 24 (left: 24). Valid! According to guidance[2]
Answer: 3 + (8 - 5) * 7 = 24

Input: 2 3 4 9
Steps:
1. Attempt 1:
   9 - 2 = 7 (left: 3, 4, 7)
   7 * 3 = 21 (left: 4, 21)
   21 + 4 = 25 (left: 25), not equal to 24  Backtrack!

2. Attempt 2:
   9 / 3 = 3 (left: 2, 4, 3)
   4 * 3 = 12 (left: 2, 12)
   12 * 2 = 24 (left: 24) According to guidance[1]
Answer: (9 / 3) * 4 * 2 = 24

Input: {QUESTION}
Steps:
Answer:
"""

ZERO_SHOT_FORMAT_TEMPLATE = """Your response should follow this format:
Answer: [mathematical expression] = 24

For example:

Answer: (7 - 5) * (7 + 5) = 24
Answer: 4 * 8 - 6 - 2 = 24
Answer: 7 * (4 - 1) + 2 = 24

Previous response: {prev_response}
Please reformat your response:"""

FEW_SHOTS_FORMAT_TEMPLATE = """Your response should follow this format:

Steps:
[...]
Answer: [mathematical expression] = 24

For example:

Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24

Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24

Steps:
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24

Previous response: {prev_response}
Please reformat your response as the format above:"""

OURS_FORMAT_TEMPLATE = """Your response should follow this format:

Reasonings:
[original reasoning process]
Answer: [mathematical expression] = 24

For example:

Reasonings:
···
Answer: 4 * (9 - (13 - 10)) = 24

Reasonings:
···
Answer: (1 + 8 / 4) * 8 = 24

Reasonings:
···
Answer: ((5 + 5) + 5) + 9 = 24

Previous response: {prev_response}
Please reformat your response as the format above:"""