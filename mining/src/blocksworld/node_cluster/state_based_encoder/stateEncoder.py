import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def check_pattern(string):
    patterns = {
        "pick": r'^pick up the (.+)$',
        "unstack": r'^unstack the (.+) from on top of the (.+)$',
        "putdown": r'^put down the (.+)$',
        "stack": r'^stack the (.+) on top of the (.+)$'
    }
    colors = {
        "a": "red block", "b": "blue block", "c": "orange block", "d": "yellow block",
        "e": "white block", "f": "magenta block", "g": "black block", "h": "cyan block",
        "i": "green block", "j": "violet block", "k": "silver block", "l": "gold block"
    }
    for action, pattern in patterns.items():
        match = re.match(pattern, string)
        if match:
            matched_colors = match.groups()
            if all(color in colors.values() for color in matched_colors):
                return True
    return False


def discrete_init_state(text):
    state = {'hand': 'empty'}
    parts = text.split(', ')
    parts = parts[0:-1] + [i.strip() for i in parts[-1].split("and")]
    for part in parts:
        if 'table' in part:
            block = part.split(" ")[1]
            state[block] = 'table'
        elif 'is on top of' in part:
            block1, block2 = part.split(' is on top of ')
            block1 = block1.split(" ")[1]
            block2 = block2.split(" ")[1]
            state[block1] = block2
    return state

def discrete_goal_state(text):
    state = {}
    parts = text.split(', ')
    parts = parts[0:-1] + [i.strip() for i in parts[-1].split("and")]
    colors = set()
    total_colors = set()
    for part in parts:
        if 'is on top of' in part:
            block1, block2 = part.split(' is on top of ')
            block1 = block1.split(" ")[1]
            block2 = block2.split(" ")[1]
            state[block1] = block2
            colors.add(block1)
            total_colors.add(block1)
            total_colors.add(block2)
    return state

def calculate_similarity(current_state, goal_state):    
    goal_vector = []
    current_vector = []
    for block in goal_state:
        goal_vector.append(1)
        current_vector.append(1 if current_state.get(block) == goal_state[block] else 0)
    return "".join(str(i) for i in current_vector)


def execute_action(state, action):
    action_parts = action.split(" ")
    
    def is_valid_action(state, action_parts):
        if action_parts[0] == 'pick':
            block = action_parts[-2]
            
            flag = True
            for key, value in state.items():
                if value == block:
                    flag = False
                    break
            return state['hand'] == 'empty' and flag and state.get(block) == 'table'
        elif action_parts[0] == 'put':
            block = action_parts[-2]
            return state['hand'] == block and state.get(block) == 'hand'
        elif action_parts[0] == 'stack':
            block = action_parts[2]
            destination_block = action_parts[-2]
            
            flag = True
            for key, value in state.items():
                if value == destination_block:
                    flag = False
                    break
            return state['hand'] == block and state.get(block) == 'hand' and flag
        elif action_parts[0] == 'unstack':
            block = action_parts[2]
            destination_block = action_parts[-2]
            
            flag = True
            for key, value in state.items():
                if value == block:
                    flag = False
                    break
            return state['hand'] == 'empty' and state.get(block) == destination_block and flag
        return False

    if not is_valid_action(state, action_parts):
        return -2

    
    
    if action_parts[0] == 'pick':
        block = action_parts[-2]
        state['hand'] = block
        state[block] = 'hand'
    
    
    elif action_parts[0] == 'put':
        block = action_parts[-2]
        state['hand'] = 'empty'
        state[block] = 'table'
    
    
    elif action_parts[0] == 'stack':
        block = action_parts[2]
        destination_block = action_parts[-2]
        state['hand'] = 'empty'
        state[block] = destination_block
    
    
    elif action_parts[0] == 'unstack':
        block = action_parts[2]
        destination_block = action_parts[-2]
        state['hand'] = block
        state[block] = 'hand'
    return state

def state_encoder(init_text: str, goal_text: str, plans: list, actions: list) -> str:
    initial_state = discrete_init_state(init_text)
    
    current_state = initial_state.copy()
    for plan in plans:
        current_state = execute_action(current_state, plan)
    goal_state = current_state
    current_state = initial_state.copy()
    if actions == [""]:
        return calculate_similarity(initial_state, goal_state)
    for action in actions:
        if check_pattern(action):
            current_state = execute_action(current_state, action)
            if current_state == -2:
                return -2
            similarity = calculate_similarity(current_state, goal_state)
        else:
            return -1
    return similarity

def visualize_state(state):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 20)  
    ax.set_ylim(0, 12)  
    ax.set_aspect('equal')
    ax.axis('off')

    colors = {
        'red': 'red',
        'blue': 'blue',
        'orange': 'orange',
        'yellow': 'yellow',
        'green': 'green',
        'violet': 'violet',
        'cyan': 'cyan',
        'magenta': 'magenta',
        'black': 'black',
        'white': 'white',
        'silver': 'silver',
        'gold': 'gold'
    }
    
    table = []
    positions = {}
    table_x_start = 1
    table_y = 1
    block_size = 2
    spacing = 3

    base_blocks = [block for block, pos in state.items() if pos == 'table']
    for i, block in enumerate(base_blocks):
        x = table_x_start + i * spacing
        y = table_y
        positions[block] = (x, y)
        table.append(block)

    
    for block in table:
        tmp = block
        while True:
            flag = True
            for key, value in state.items():
                if value == tmp:
                    lower_x, lower_y = positions[value]
                    x = lower_x
                    y = lower_y + block_size + 0.1
                    positions[key] = (x, y)
                    flag = False
                    tmp = key
                    break
            if flag:
                break
    
    for block, (x, y) in positions.items():
        color = colors.get(block, 'gray')
        rect = patches.Rectangle((x, y), block_size, block_size, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + block_size / 2, y + block_size / 2, block, ha='center', va='center')
    
    hand_x_start = 17
    hand_y_start = 1
    hand_width = 2.5
    hand_height = 2.5
    hand_rect = patches.Rectangle((hand_x_start, hand_y_start), hand_width, hand_height, 
                                  linewidth=1, edgecolor='black', linestyle='--', facecolor='none')
    ax.add_patch(hand_rect)
    ax.text(hand_x_start + hand_width / 2, hand_y_start + hand_height + 0.5, 'hand', ha='center', va='center')

    if state['hand'] != 'empty':
        block_in_hand = state['hand']
        color = colors.get(block_in_hand, 'gray')
        rect = patches.Rectangle((hand_x_start + 0.25, hand_y_start + 0.25), block_size, block_size, facecolor=color)
        ax.add_patch(rect)
        ax.text(hand_x_start + hand_width / 2, hand_y_start + hand_height / 2, block_in_hand, ha='center', va='center')

    plt.show()

    


