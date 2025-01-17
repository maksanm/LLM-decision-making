from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI


class ContextDiscretizerChain:
    CONTEXT_DISCRETIZATION_PROMPT_TEMPLATE = """
TODO: write prompt after StatesEnumerationChain implementation
{states_space}
{initial_context}
"""

    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)

    def create(self):
        return {
            "contexts": PromptTemplate.from_template(self.CONTEXT_DISCRETIZATION_PROMPT_TEMPLATE)
            | self.llm
            | JsonOutputParser()
            | RunnableLambda(lambda response: response["contexts"])
        }
