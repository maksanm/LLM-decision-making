from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


class InputValidationChain:
    VALIDATION_PROMPT_TEMPLATE = """Validate if the user request contains a clear goal that requires a decision. Return 'True' if it does, otherwise return 'False'.

User request:
{user_request}

Response:
"""

    def create(self):
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
        return {
             "is_valid": PromptTemplate.from_template(self.VALIDATION_PROMPT_TEMPLATE)
                | llm
                | StrOutputParser()
                | self._parse_bool
        }

    def _parse_bool(self, str):
        if str == "True": return True
        elif str == "False": return True
        raise Exception("Unable to parse the LLM output")
