from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


class CanExpandActionsChain:
    CAN_EXPAND_PROMPT_TEMPLATE = """User request:
{user_request}

Initial actions from the user request:
{initial_actions}

Evaluate the user request to determine if there is potential to expand the action space beyond the initial actions described above. Respond 'True' only if the context of the user request explicitly suggests or necessitates additional actions that are not already covered by the initial actions. Respond 'False' if all necessary and possible actions are explicitly mentioned and covered in the initial actions provided.

Response:
"""

    def create(self):
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
        return (
            PromptTemplate.from_template(self.CAN_EXPAND_PROMPT_TEMPLATE)
            | llm
            | StrOutputParser()
        )
