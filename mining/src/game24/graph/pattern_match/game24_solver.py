import itertools

# a op c op c op d
def build_expr_no_bra(num1, num2, num3, num4, op1, op2, op3):
    return f"{num1} {op1} {num2} {op2} {num3} {op3} {num4}"

# (a op b) op c op d
def build_expr_bra1(num1, num2, num3, num4, op1, op2, op3):
    return f"({num1} {op1} {num2}) {op2} {num3} {op3} {num4}"

# (a op b op c) op d
def build_expr_bra2(num1, num2, num3, num4, op1, op2, op3):
    return f"({num1} {op1} {num2} {op2} {num3}) {op3} {num4}"

# a op (b op c) op d
def build_expr_bra3(num1, num2, num3, num4, op1, op2, op3):
    return f"{num1} {op1} ({num2} {op2} {num3}) {op3} {num4}"

# a op b op (c op d)
def build_expr_bra4(num1, num2, num3, num4, op1, op2, op3):
    return f"{num1} {op1} {num2} {op2} ({num3} {op3} {num4})"

# a op (b op c op d)
def build_expr_bra5(num1, num2, num3, num4, op1, op2, op3):
    return f"{num1} {op1} ({num2} {op2} {num3} {op3} {num4})"

# (a op b) op (c op d)
def build_expr_double_bra1(num1, num2, num3, num4, op1, op2, op3):
    return f"({num1} {op1} {num2}) {op2} ({num3} {op3} {num4})"

# a op (b op (c op d))
def build_expr_double_bra2(num1, num2, num3, num4, op1, op2, op3):
    return f"{num1} {op1} ({num2} {op2} ({num3} {op3} {num4}))"

# a op ((b op c) op d)
def build_expr_double_bra3(num1, num2, num3, num4, op1, op2, op3):
    return f"{num1} {op1} (({num2} {op2} {num3}) {op3} {num4})"

# (a op (b op c)) op d
def build_expr_double_bra4(num1, num2, num3, num4, op1, op2, op3):
    return f"({num1} {op1} ({num2} {op2} {num3})) {op3} {num4}"

# ((a op b) op c) op d
def build_expr_double_bra5(num1, num2, num3, num4, op1, op2, op3):
    return f"(({num1} {op1} {num2}) {op2} {num3}) {op3} {num4}"

# a op b op c
def build_expr_3_no_bra(num1, num2, num3, op1, op2):
    return f"{num1} {op1} {num2} {op2} {num3}"

# (a op b) op c
def build_expr_3_bra1(num1, num2, num3, op1, op2):
    return f"({num1} {op1} {num2}) {op2} {num3}"

# a op (b op c)
def build_expr_3_bra2(num1, num2, num3, op1, op2):
    return f"{num1} {op1} ({num2} {op2} {num3})"

def build_expr_2(num1, num2, op):
    return f"{num1} {op} {num2}"

def calc_expr(num1, num2, num3, num4, op1, op2, op3, func):
    try:
        if abs(eval(func(num1, num2, num3, num4, op1, op2, op3)) - 24) < 1e-3:
            return func(num1, num2, num3, num4, op1, op2, op3)
        return False
    except ZeroDivisionError:
        return False
    
def calc_expr_3(num1, num2, num3, op1, op2, func):
    try:
        if abs(eval(func(num1, num2, num3, op1, op2)) - 24) < 1e-3:
            return func(num1, num2, num3, op1, op2)
        return False
    except ZeroDivisionError:
        return False
    
def calc_expr_2(num1, num2, op, func):
    try:
        if abs(eval(func(num1, num2, op)) - 24) < 1e-3:
            return func(num1, num2, op)
        return False
    except ZeroDivisionError:
        return False
    
def calc_exprs(num1, num2, num3, num4):
    acc = []
    nums = (num1, num2, num3, num4)
    ops = ('+', '-', '*', '/')
    for num_perm in itertools.permutations(nums):
        for op_perm in itertools.product(ops, ops, ops):
            for func in [build_expr_no_bra, build_expr_bra1, build_expr_bra2, build_expr_bra3, build_expr_bra4, build_expr_bra5, build_expr_double_bra1, build_expr_double_bra2, build_expr_double_bra3, build_expr_double_bra4, build_expr_double_bra5]:
                expr = calc_expr(*num_perm, *op_perm, func)
                if expr:
                    acc.append(expr)
    
    return acc

def calc_exprs_4(num1, num2, num3, num4):
    nums = (num1, num2, num3, num4)
    ops = ('+', '-', '*', '/')
    for num_perm in itertools.permutations(nums):
        for op_perm in itertools.permutations(ops, 3):
            for func in [build_expr_no_bra, build_expr_bra1, build_expr_bra2, build_expr_bra3, build_expr_bra4, build_expr_bra5, build_expr_double_bra1, build_expr_double_bra2, build_expr_double_bra3, build_expr_double_bra4, build_expr_double_bra5]:
                if calc_expr(*num_perm, *op_perm, func):
                    return True
    return False

def calc_exprs_3(num1, num2, num3):
    nums = (num1, num2, num3)
    ops = ('+', '-', '*', '/')
    for num_perm in itertools.permutations(nums):
        for op_perm in itertools.permutations(ops, 2):
            for func in [build_expr_3_no_bra, build_expr_3_bra1, build_expr_3_bra2]:
                if calc_expr_3(*num_perm, *op_perm, func):
                    return True
    return False

def calc_exprs_2(num1, num2):
    ops = ('+', '-', '*', '/')
    for op in ops:
        if calc_expr_2(num1, num2, op, build_expr_2) or calc_expr_2(num2, num1, op, build_expr_2):
            return True
    return False

def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0].split(' ')