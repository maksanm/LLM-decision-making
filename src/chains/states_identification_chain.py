from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


class StatesIdentificationChain:
    STATES_IDENTIFICATION_PROMPT_TEMPLATE = """
"""

    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)


    def create(self):
        return RunnablePassthrough()