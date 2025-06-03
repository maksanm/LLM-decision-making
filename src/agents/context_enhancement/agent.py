from .chains.context_completion_chain import ContextCompletionChain
from .chains.context_discretization_chain import ContextDiscretizationChain


class ContextEnhancementAgent:

    def __init__(self):
        self.context_discritization_chain = ContextDiscretizationChain().create()
        self.context_completion_chain = ContextCompletionChain().create()


    def invoke(self, state):
        if state["initial_context"] != "":
            state = state | self.context_discritization_chain.invoke(state)
        else:
            state = state | { "contexts": ["" for _ in state["state_space"]] }
        enhanced_contexts = self.context_completion_chain.invoke(state)
        return {
            "contexts" : enhanced_contexts
        }