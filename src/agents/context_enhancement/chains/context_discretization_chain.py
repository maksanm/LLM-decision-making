from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI


class ContextDiscretizationChain:
    CONTEXT_DISCRETIZATION_PROMPT_TEMPLATE = """
TODO: write prompt after StatesEnumerationAgent implementation
{states_space}
{initial_context}
Return JSON {{"contexts": []}}
"""

    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0.0)


    def create(self):
        return (
            PromptTemplate.from_template(self.CONTEXT_DISCRETIZATION_PROMPT_TEMPLATE)
            | self.llm
            | JsonOutputParser()
        )
