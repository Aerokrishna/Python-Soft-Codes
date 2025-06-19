# bt_core/base.py

SUCCESS = True
FAILURE = False

class BTNode:
    def __init__(self):
        self.completed = False  # This flag determines node success

    def run(self, blackboard):
        self.run_logic(blackboard)
        return SUCCESS if self.completed else FAILURE

    def run_logic(self, blackboard):
        raise NotImplementedError("Override run_logic in subclass")
