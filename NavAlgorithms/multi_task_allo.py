import numpy as np
import matplotlib.pyplot as plt

heading_dir = "CW"

numboxes = 6

boxes_x = np.random.uniform(10, 470, (numboxes,)).astype(int)
boxes_y = np.random.uniform(10, 640, (numboxes,)).astype(int)
boxes = np.column_stack((boxes_x, boxes_y))

stack = np.array([320,240])

# robots = np.array([[160, 120], [360, 120], [160, 360], [360, 360], [160, 400]])

robot_x = np.random.uniform(10, 470, (5,)).astype(int)
robot_y = np.random.uniform(10, 640, (5,)).astype(int)
robots = np.column_stack((robot_x, robot_y))


def compute_vector(x1, x2, y1, y2):
    return np.arctan2((y2 - y1), (x2 - x1))

def compute_distance(x1, x2, y1, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def normalize_cost(cost1, cost2, w1, w2):
    angle_min, angle_max = min(cost1), max(cost1)
    dist_min, dist_max = min(cost2), max(cost2)

    normalized_scores = []
    for i in range(len(boxes)):
        angle_norm = (cost1[i] - angle_min) / (angle_max - angle_min + 1e-8)
        dist_norm = (cost2[i] - dist_min) / (dist_max - dist_min + 1e-8)

        # weighted sum: weights can be tuned
        total_cost = w1 * angle_norm + w2 * dist_norm
        normalized_scores.append((boxes[i], total_cost))

    return normalized_scores

def get_optimal_box(boxes, bots, stack):
    optimal_boxes = []
    reserved = []

    for robot in bots:

        angle_costs = []
        dist_costs = []

        acw = []
        cw = []
        angle_stack = compute_vector(robot[0], stack[0], robot[1], stack[1])

        for box in boxes:
            angle_box = compute_vector(robot[0], box[0], robot[1], box[1])
            bot_to_box = compute_distance(robot[0], box[0], robot[1], box[1])
            box_to_stack = compute_distance(stack[0], box[0], stack[1], box[1])

            angle_diff = np.arctan2(np.sin(angle_box - angle_stack), np.cos(angle_box - angle_stack))


            # if heading direction is clockwise we need the angle difference to be more negative
            angle_cost =  -np.rad2deg(angle_diff)
            dist_cost = bot_to_box + box_to_stack

            angle_costs.append(angle_cost)
            dist_costs.append(dist_cost)

            if angle_diff > 0:
                # print("BOX :", box, "CW HEADING angle_cOST :", angle_cost)
                # print("BOX :", box, "Dist cost :", dist_cost)

                cw.append(box)
            else:
                # print("BOX :", box, "ACW HEADING angle_cOST :", angle_cost)
                # print("BOX :", box, "Dist Cost :", dist_cost)

                acw.append(box)

        normalized_cost = normalize_cost(angle_costs, dist_costs, 0.5, 0.5)
        # Choose the best box
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

# Make a copy of boxes for assignment logic
boxes_for_assignment = boxes.tolist()
optimal_boxes = get_optimal_box(boxes_for_assignment, robots, stack)

plt.figure(figsize=(8, 6))

# Plot stack (red)
plt.scatter(stack[0], stack[1], color='red', s=100, label='Stack')

# Plot all boxes: assigned (orange), unassigned (green)
assigned_boxes = [tuple(best_box) for _, best_box in optimal_boxes]
for box in boxes:
    if tuple(box) in assigned_boxes:
        plt.scatter(box[0], box[1], color='orange', s=100, label='Assigned Box')
    else:
        plt.scatter(box[0], box[1], color='green', s=50, label='Unassigned Box')

# Plot robots and arrows
for robot, best_box in optimal_boxes:       
    plt.scatter(robot[0], robot[1], color='blue', s=100, label='Robot')
    # Draw arrow from robot to its best box
    plt.arrow(robot[0], robot[1], best_box[0]-robot[0], best_box[1]-robot[1],
            head_width=5, head_length=10, fc='black', ec='black', length_includes_head=True)

plt.xlim(0, 500)
plt.ylim(0, 700)
plt.xlabel('X')
plt.ylabel('Y')
# Remove duplicate labels in legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.title('Robot, Stack, and Boxes')
plt.grid(True)
plt.show()