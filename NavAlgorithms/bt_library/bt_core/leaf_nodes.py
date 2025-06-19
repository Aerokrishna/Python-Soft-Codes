# bt_core/leaf_nodes.py

from bt_core.base import BTNode

class ConditionNode(BTNode):
    def __init__(self, condition_func):
        super().__init__()
        self.condition_func = condition_func

    def run_logic(self, blackboard):
        self.completed = self.condition_func(self, blackboard)


class ActionNode(BTNode):
    def __init__(self, action_func):
        super().__init__()
        self.action_func = action_func

    def run_logic(self, blackboard):
        self.completed = self.action_func(self, blackboard)
