from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from perplexity_reriever import PerplexityRetriever
from tavily_retriever import TavilyRetriever
from langchain_core.runnables import RunnablePassthrough


class WebRetriever:
    AGGREGATION_PROMPT_TEMPLATE = """Using the following data:

<PERPLEXITY-RESPONSE>
{perplexity_response}
</PERPLEXITY-RESPONSE>

<TAVILY-RESPONSE>
{tavily_response}
</TAVILY-RESPONSE>

Please provide a summary that integrates information from both sources to respond the below query:
{input}
"""

    def __init__(self, domains=None):
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
        self.perplexity_retriever = PerplexityRetriever(domains=domains)
        self.tavily_retriever = TavilyRetriever(domains=domains)


    def create(self):
        return (
            RunnablePassthrough.assign(
                perplexity_response=lambda state: self.perplexity_retriever.invoke(state["input"]),
                tavily_response=lambda state: self.tavily_retriever.invoke(state["input"])
            )
            | PromptTemplate.from_template(self.AGGREGATION_PROMPT_TEMPLATE)
            | self.llm
            | StrOutputParser()
        )
