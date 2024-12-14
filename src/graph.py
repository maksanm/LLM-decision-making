from chains.initial_analysis_chain import InitialAnalysisChain
from chains.action_expansion_chain import ActionExpansionChain

from langgraph.graph import END, StateGraph
from typing import Annotated, TypedDict, List
from operator import add


class DecisionState(TypedDict):
    user_request: str
    goal_definition: str
    initial_context: str
    action_space: Annotated[List[str], add]


class DecisionGraph:

    def create(self):
        graph = StateGraph(DecisionState)

        graph.add_node("initial_analysis_agent", InitialAnalysisChain().create())
        graph.add_node("action_expansion_agent", ActionExpansionChain().create())

        graph.add_edge("initial_analysis_agent", "action_expansion_agent")
        graph.add_edge("action_expansion_agent", END)

        graph.set_entry_point("initial_analysis_agent")

        return graph.compile()
