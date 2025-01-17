from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnablePassthrough
import os

from chains.web_search_chain import WebSearchChain


class ActionsExpansionChain:
    CAN_EXPAND_PROMPT_TEMPLATE = """User request:
{user_request}

Goal definition from the user request:
{goal_definition}

Initial actions from the user request:
{initial_actions}

Evaluate the user request to determine if there is potential to expand the action space beyond the initial actions described above. Respond 'TRUE' only if the context or goal within the user request explicitly suggests or necessitates additional actions that are not already covered by the initial actions. Respond 'FALSE' if all necessary and possible actions are explicitly mentioned and covered in the initial actions provided.

Response:
"""

    ACTION_EXPANSION_PROMPT_TEMPLATE = """Goal Definition:
{goal_definition}

Initial actions:
{initial_actions}

Retrieved data:
<RETRIEVED_DATA>
{retrieved_data}
</RETRIEVED_DATA>

Analyze the retrieved data to identify up to """+ os.getenv("EXPANDED_ACTIONS_MAX_LIMIT") + """ of the most reliable additional actions that can be taken to achieve the goal stated above. These actions should be distinct yet similar to the initial ones, maintaining strong relevance and consistency within the same domain (e.g., if deciding on crops to plant next year, suggest other specific crop options). Each action must be concrete, directly applicable, and executable, leaving no room for vague or abstract considerations. Ensure these additional actions are directly comparable to the initial ones. Provide only what to do, without explanations, in the following JSON format:
{{
    "new_actions": ["Action a", "Action b", ...]
}}

Response:
"""

    def create(self):
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
        return (
            RunnableLambda(lambda state: self._expand_state_with_initial_actions(state))
            | RunnablePassthrough.assign(
                can_expand=PromptTemplate.from_template(self.CAN_EXPAND_PROMPT_TEMPLATE)
                | llm
                | StrOutputParser(),
                search_query=lambda state: f"Goal Definition: {state["goal_definition"]} Initially Proposed Actions: {state["initial_actions"]}. Please suggest proposed actions alternetives."
            )
            | RunnableBranch(
                (lambda state: state["can_expand"] == "FALSE", {}),
                {
                    "action_space": RunnablePassthrough().assign(
                        retrieved_data=WebSearchChain().create()
                    )| RunnablePassthrough.assign(zx=lambda x: print(x))
                    | PromptTemplate.from_template(self.ACTION_EXPANSION_PROMPT_TEMPLATE)
                    | llm
                    | JsonOutputParser()
                    | RunnableLambda(lambda actions_dict: actions_dict["new_actions"])
                }
            )
        )

    def _expand_state_with_initial_actions(self, state):
        if len(state["action_space"]) > 0:
            state["initial_actions"] = self._get_initial_actions_string(state["action_space"])
        else:
            state["initial_actions"] = "No initial actions specified."
        return state

    def _get_initial_actions_string(self, initial_actions):
        return ", ".join(initial_actions)
