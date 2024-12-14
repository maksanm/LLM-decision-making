from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnablePassthrough
from langchain_community.retrievers import TavilySearchAPIRetriever


class ActionExpansionChain:
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

Analyze the retrieved data to identify the most reliable additional actions that can be taken to achieve the goal stated above. These actions should be distinct from, but similar to, the initial actions, maintaining strong relevance and consistency within the same domain (e.g., if deciding on crops to plant next year, suggest other specific crop options). Ensure that these additional actions are directly comparable to the initial ones. Provide the additional actions in the following JSON format:
{{
    "new_actions": ["Action a", "Action b", ...]
}}

Response:
"""

    def create(self):
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
        retriever = TavilySearchAPIRetriever(
            k=4,
            include_generated_answer=True,
            include_domains=["https://www.usda.gov/"], # Use predifined data sources specific to the decision domain
            search_depth="advanced"
        )
        return (
            RunnableLambda(lambda state: self._expand_state_with_initial_actions(state))
            | RunnablePassthrough.assign(
                can_expand=PromptTemplate.from_template(self.CAN_EXPAND_PROMPT_TEMPLATE)
                | llm
                | StrOutputParser()
            )
            | RunnableBranch(
                (lambda state: state["can_expand"] == "FALSE", {}),
                {
                    "action_space": RunnablePassthrough.assign(
                        retrieved_data=RunnableLambda(lambda state: state["goal_definition"])
                            | retriever
                            | self._format_tavily_response,
                    )
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

    def _format_tavily_response(self, retrieved_documents):
        formatted_response = []
        for document in retrieved_documents:
            content = document.page_content
            formatted_response.append(f"{content}\n")
        return "\n".join(formatted_response)
