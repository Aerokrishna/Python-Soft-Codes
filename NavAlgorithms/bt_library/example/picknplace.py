# example/pick_and_place.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bt_core import Sequence, ActionNode

# Simple shared blackboard
blackboard = {
    'arm_pose': 'home',
    'object_picked': False
}

# Action: move to object
def move_to_object(self, bb):
    print("Moving to object...")
    if bb['arm_pose'] == 'object':
        print("Already at object.")
        return True
    bb['arm_pose'] = 'object'
    return True

# Action: pick object
def pick(self, bb):
    print("Picking object...")
    if not bb['object_picked']:
        bb['object_picked'] = True
        return True
    return False

# Action: move to drop location
def move_to_drop(self, bb):
    print("Moving to drop location...")
    if bb['arm_pose'] == 'drop':
        print("Already at drop location.")
        return True
    bb['arm_pose'] = 'drop'
    return True

# Action: place object
def place(self, bb):
    print("Placing object...")
    if bb['object_picked']:
        bb['object_picked'] = False
        return True
    return False

# Build the behavior tree
pick_sequence = Sequence([
    ActionNode(move_to_object),
    ActionNode(pick)
])

place_sequence = Sequence([
    ActionNode(move_to_drop),
    ActionNode(place)
])

main_bt = Sequence([
    pick_sequence,
    place_sequence
])

# Run
print("\n--- Executing Pick and Place Behavior Tree ---\n")
main_bt.run(blackboard)
