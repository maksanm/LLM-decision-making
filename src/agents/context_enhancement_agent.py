from chains.context_completion_chain import ContextCompletionChain
from chains.context_discretization_chain import ContextDiscretizerChain


class ContextEnhancementAgent:

    def __init__(self):
        self.context_discritizer_chain = ContextDiscretizerChain().create()
        self.context_completion_chain = ContextCompletionChain().create()


    def invoke(self, state):
        if state["initial_context"] != "":
            return {}
        else:
            return {}