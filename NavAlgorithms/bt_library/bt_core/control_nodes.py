# bt_core/control_nodes.py

from bt_core.base import BTNode, SUCCESS, FAILURE

class Sequence(BTNode):
    def __init__(self, children):
        super().__init__()
        self.children = children

    def run(self, blackboard):
        for child in self.children:
            if child.run(blackboard) == FAILURE:
                self.completed = False
                return FAILURE
        self.completed = True
        return SUCCESS

class Fallback(BTNode):
    def __init__(self, children):
        super().__init__()
        self.children = children

    def run(self, blackboard):
        for child in self.children:
            if child.run(blackboard) == SUCCESS:
                self.completed = True
                return SUCCESS
        self.completed = False
        return FAILURE
