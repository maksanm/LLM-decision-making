from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class StatesEnumerationChain:
    _PROMPT_TEMPLATE = """
"""

    def create(self):
        return RunnablePassthrough()