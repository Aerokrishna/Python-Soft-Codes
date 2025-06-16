import numpy as np
import matplotlib.pyplot as plt

heading_dir = "CW"
numboxes = 12

# Generate random box and robot positions
boxes_x = np.random.uniform(10, 470, (numboxes,)).astype(int)
boxes_y = np.random.uniform(10, 640, (numboxes,)).astype(int)
boxes = np.column_stack((boxes_x, boxes_y))

stack = np.array([320, 240])
tower = [["a", "b", "c"], ["d", "e"], ["f"]]

robot_x = np.random.uniform(10, 470, (5,)).astype(int)
robot_y = np.random.uniform(10, 640, (5,)).astype(int)
robots = np.column_stack((robot_x, robot_y))

# ------------------------ Utility Functions ------------------------

def compute_vector(x1, x2, y1, y2):
    return np.arctan2((y2 - y1), (x2 - x1))

def compute_distance(x1, x2, y1, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def normalize_cost(cost1, cost2, w1, w2):
    angle_min, angle_max = min(cost1), max(cost1)
    dist_min, dist_max = min(cost2), max(cost2)

    normalized_scores = []
    for i in range(len(boxes)):
        angle_norm = (cost1[i] - angle_min) / (angle_max - angle_min + 1e-8)
        dist_norm = (cost2[i] - dist_min) / (dist_max - dist_min + 1e-8)
        total_cost = w1 * angle_norm + w2 * dist_norm
        normalized_scores.append((boxes[i], total_cost))
    return normalized_scores

# ------------------------ Box Assignment ------------------------

def get_optimal_box(boxes, bots, stack):
    optimal_boxes = []
    reserved = []

    for robot in bots:
        angle_costs, dist_costs = [], []
        angle_stack = compute_vector(robot[0], stack[0], robot[1], stack[1])

        for box in boxes:
            angle_box = compute_vector(robot[0], box[0], robot[1], box[1])
            bot_to_box = compute_distance(robot[0], box[0], robot[1], box[1])
            box_to_stack = compute_distance(stack[0], box[0], stack[1], box[1])

            angle_diff = np.arctan2(np.sin(angle_box - angle_stack), np.cos(angle_box - angle_stack))
            angle_cost = -np.rad2deg(angle_diff)
            dist_cost = bot_to_box + box_to_stack

            angle_costs.append(angle_cost)
            dist_costs.append(dist_cost)

        normalized_cost = normalize_cost(angle_costs, dist_costs, 0.5, 0.5)
        normalized_cost.sort(key=lambda x: x[1])

        for i in range(len(normalized_cost)):
            box_tuple = tuple(normalized_cost[i][0])
            if box_tuple not in reserved:
                best_box = normalized_cost[i][0]
                optimal_boxes.append((robot, best_box))
                reserved.append(box_tuple)
                break

        print(f"Best box to pick for bot {robot}:", best_box)

    return optimal_boxes

# ------------------------ Tower Spot Assignment ------------------------

def create_tower(stack):
    sx, sy = stack
    offset_x = 30
    offset_y = 30

    layer_1_dis = 20
    layer_2_dis = 30

    layer1 = [np.array([sx - offset_x, sy]),
              np.array([sx, sy]),
              np.array([sx + offset_x, sy])]
    layer2 = [np.array([sx - offset_x / 2, sy]),
              np.array([sx + offset_x / 2, sy])]
    layer3 = [np.array([sx, sy - 2 * offset_y])]

    return [layer1, layer2, layer3]

def select_best_spot(bot, spots, box):
    best_spot = None
    best_cost = float('inf')
    for spot in spots:
        cost = compute_distance(bot[0], spot[0], bot[1], spot[1]) + compute_distance(bot[0], box[0], bot[1], box[1])
        if cost < best_cost:
            best_cost = cost
            best_spot = spot
    return best_spot

def assign_tower_spots(optimal_boxes, stack):
    side_a = [entry for entry in optimal_boxes if entry[1][1] < stack[1]]
    side_b = [entry for entry in optimal_boxes if entry[1][1] >= stack[1]]

    boxes = [box for box in optimal_boxes]

    tower_layers = create_tower(stack)
    assignments = []
    used_spots = []

    turn = 0  # Start with side A

    for layer in tower_layers:
        remaining_spots = layer.copy()
        while remaining_spots:
            if not boxes:
                break
            bot, box = boxes.pop(0)
            # if turn % 2 == 0 and side_a:
            #     bot, box = side_a.pop(0)
            # elif side_b:
            #     bot, box = side_b.pop(0)
            # else:
            #     break

            available = [s for s in remaining_spots if tuple(s) not in [tuple(u) for u in used_spots]]
            if not available:
                break

            best_spot = select_best_spot(bot, available, box)
            assignments.append((bot, box, best_spot))
            used_spots.append(best_spot)
            # Remove best_spot using np.array_equal to avoid ValueError
            remaining_spots = [s for s in remaining_spots if not np.array_equal(s, best_spot)]

            turn += 1

    return assignments

# ------------------------ Main Flow ------------------------

boxes_for_assignment = boxes.tolist()
optimal_boxes = get_optimal_box(boxes_for_assignment, robots, stack)
assignments = assign_tower_spots(optimal_boxes, stack)

# ------------------------ Visualization ------------------------

plt.figure(figsize=(8, 6))

# Plot stack (red)
plt.scatter(stack[1], stack[0], color='red', s=100, label='Stack')

# Plot boxes
assigned_boxes = [tuple(box) for _, box in optimal_boxes]
for box in boxes:
    color = 'orange' if tuple(box) in assigned_boxes else 'green'
    plt.scatter(box[1], box[0], color=color, s=80)

# Plot full robot → box → stack assignment
for robot, box, spot in assignments:
    plt.scatter(robot[1], robot[0], color='blue', s=100, label='Robot')
    plt.arrow(robot[1], robot[0], box[1] - robot[1], box[0] - robot[0],
              head_width=4, head_length=6, fc='black', ec='black', length_includes_head=True)
    plt.arrow(box[1], box[0], spot[1] - box[1], spot[0] - box[0],
              head_width=4, head_length=6, fc='darkorange', ec='darkorange', length_includes_head=True)
    plt.scatter(spot[1], spot[0], color='purple', s=100, label='Tower Spot')

# Clean up legend
handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))
plt.legend(unique.values(), unique.keys())
plt.xlim(0, 700)
plt.ylim(0, 500)
plt.xlabel('Y')
plt.ylabel('X')
plt.grid(True)
plt.title("Robot Picks and Tower Drop Assignments (X/Y Flipped)")
plt.show()
