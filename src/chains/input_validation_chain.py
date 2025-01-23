from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


# Needed to support the chatbot experience in the future
class InputValidationChain:
    VALIDATION_PROMPT_TEMPLATE = """Validate if the user request contains a clear goal that requires a decision. Return 'True' if it does, otherwise return 'False'.

User request:
{user_request}

Response:
"""


    def create(self):
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
        return (
            PromptTemplate.from_template(self.VALIDATION_PROMPT_TEMPLATE)
            | llm
            | StrOutputParser()
        )