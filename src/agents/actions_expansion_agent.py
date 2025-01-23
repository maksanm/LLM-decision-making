from chains.can_expand_actions_chain import CanExpandActionsChain
from chains.actions_expansion_chain import ActionsExpansionChain
from chains.web_search_chain import WebSearchChain


class ActionsExpansionAgent:

    def __init__(self):
        self.web_search_chain = WebSearchChain().create()
        self.can_expand_actions_chain = CanExpandActionsChain().create()
        self.actions_expansion_chain = ActionsExpansionChain().create()


    def invoke(self, state):
        state = self._expand_state_with_initial_actions(state)

        can_expand_str = self.can_expand_actions_chain.invoke(state)
        can_expand = self._parse_bool(can_expand_str)
        if not can_expand:
            return {}

        search_query = f"User request was \"{state["user_request"]}\", so initially proposed actions are {state["initial_actions"]}. Please suggest proposed actions alternetives."
        state = state | {"search_query": search_query}
        state = state | {"retrieved_data": self.web_search_chain.invoke(state)}

        try:
            response = self.actions_expansion_chain.invoke(state)
            return {
                "action_space": response["new_actions"],
                "expanded_actions": response["new_actions"]
            }
        except Exception as e:
            print(f"Error parsing the expanded actions: {e}")
            raise e


    def _expand_state_with_initial_actions(self, state):
        if len(state["action_space"]) > 0:
            return state | {"initial_actions": self._get_initial_actions_string(state["action_space"])}
        else:
            state["initial_actions"] = "No initial actions specified."
        return state


    def _get_initial_actions_string(self, initial_actions):
        return ", ".join(initial_actions)


    def _parse_bool(self, str):
        if str == "True": return True
        elif str == "False": return True
        raise Exception("Unable to parse the LLM output")
