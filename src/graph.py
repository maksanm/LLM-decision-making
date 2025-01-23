
from langgraph.graph import END, StateGraph
from typing import Annotated, TypedDict, List
from operator import add

from agents.initial_analysis_agent import InitialAnalysisAgent
from agents.actions_expansion_agent import ActionsExpansionAgent
from agents.states_enumeration_agent import StatesEnumerationAgent
from agents.context_enhancement_agent import ContextEnhancementAgent


class DecisionState(TypedDict):
    user_request: str
    goal_definition: str
    initial_context: str
    action_space: Annotated[List[str], add]
    is_valid: bool
    expanded_actions: str


class DecisionGraph:

    def _validation_routing(self, state):
        if state["is_valid"] == True:
            return ["action_expansion_agent", "states_enumeration_agent"]
        else:
            return END


    def create(self):
        graph = StateGraph(DecisionState)

        graph.add_node("initial_analysis_agent", InitialAnalysisAgent().invoke)
        graph.add_node("action_expansion_agent", ActionsExpansionAgent().invoke)
        graph.add_node("states_enumeration_agent", StatesEnumerationAgent().invoke)
        graph.add_node("context_enhancement_agent", ContextEnhancementAgent().invoke)

        graph.add_conditional_edges("initial_analysis_agent", self._validation_routing)
        graph.add_edge("action_expansion_agent", "context_enhancement_agent")
        graph.add_edge("states_enumeration_agent", "context_enhancement_agent")
        graph.add_edge("context_enhancement_agent", END)

        graph.set_entry_point("initial_analysis_agent")

        return graph.compile()
