
from langgraph.graph import END, StateGraph
from typing import Annotated, TypedDict, List
from operator import add

from chains.actions_expansion_chain import ActionsExpansionChain
from chains.context_completion_chain import ContextsCompletionChain
from chains.context_discretizer_chain import ContextDiscretizerChain
from chains.initial_analysis_chain import InitialAnalysisChain
from chains.input_validation_chain import InputValidationChain


class DecisionState(TypedDict):
    user_request: str
    goal_definition: str
    initial_context: str
    action_space: Annotated[List[str], add]
    is_valid: bool


class DecisionGraph:

    def _validation_routing(self, state):
        if state["is_valid"] == True:
            return "initial_analysis_agent"
        else:
            return END

    def _context_enhancement_routing(self, state):
        if state["initial_context"] != "":
            return "context_discretizer_agent"
        else:
            return "contexts_completion_agent"

    def create(self):
        graph = StateGraph(DecisionState)

        graph.add_node("input_validation_agent", InputValidationChain().create())
        graph.add_node("initial_analysis_agent", InitialAnalysisChain().create())
        graph.add_node("action_expansion_agent", ActionsExpansionChain().create())
        graph.add_node("context_discretizer_agent", ContextDiscretizerChain().create())
        #graph.add_node("contexts_completion_agent", ContextsCompletionChain().create())

        graph.add_conditional_edges("input_validation_agent", self._validation_routing)
        graph.add_conditional_edges("initial_analysis_agent", self._context_enhancement_routing)
        graph.add_edge("initial_analysis_agent", "action_expansion_agent")
        graph.add_edge("action_expansion_agent", "context_discretizer_agent")
        graph.add_edge("context_discretizer_agent", END)
        #graph.add_edge("contexts_completion_agent", END)


        graph.set_entry_point("input_validation_agent")

        return graph.compile()
