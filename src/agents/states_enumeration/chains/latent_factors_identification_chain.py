from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import os

from retrievers.web_retriever.retriever import WebRetriever

class LatentFactorsIdentificationChain:
    LATENT_FACTORS_IDENTIFICATION_PROMPT_TEMPLATE = """User request:
{user_request}

Goal definition:
{goal_definition}

Retrieved data:
<RETRIEVED-DATA>
{retrieved_data}
</RETRIEVED-DATA>

Based on the retrieved data, select a list of unknown latent factors and states of nature that can affect the decision process. The states of nature should describe specific events or circumstances that may happen and impact the decision-making process. Use clear and descriptive statements that reflect potential future scenarios. For example, states could include events like "The market demand will significantly increase" or "A new competitor enters the market".

Select a maximum of """ + os.getenv("LATENT_FACTORS_MAX_LIMIT") + """ states. Be concise in descriptions.

Respond in the following JSON format:

```json
{{
    "latent_factors": ["State 1 description of a possible event", "State 2 description of another possible event", "..."]
}}
```"""

    def __init__(self):
        self.web_retriever = WebRetriever().create()

    def create(self):
        self.llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0.0)
        return (
            RunnablePassthrough.assign(
                retrieved_data=lambda state: self._retrieve_data(state)
            )
            | PromptTemplate.from_template(self.LATENT_FACTORS_IDENTIFICATION_PROMPT_TEMPLATE)
            | self.llm
            | JsonOutputParser()
        )

    def _retrieve_data(self, state):
        web_search_query = f"The user request is:\n`{state["user_request"]}`\nAnalyze current online data for risks (e.g., weather, logistics, safety etc.) that could impact the user's goal: '{state['goal_definition']}', and highlight key threats."
        return self.web_retriever.invoke({"input": web_search_query})
