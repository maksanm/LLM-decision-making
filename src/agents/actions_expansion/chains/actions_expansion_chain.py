from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import os

from retrievers.web_retriever.retriever import WebRetriever


class ActionsExpansionChain:
    ACTION_EXPANSION_PROMPT_TEMPLATE = """Goal Definition:
{goal_definition}

Initial actions:
{initial_actions}

Retrieved data:
```
{retrieved_data}
```

Analyze the retrieved data to identify up to """+ os.getenv("EXPANDED_ACTIONS_MAX_LIMIT") + """ of the most reliable additional actions that can be taken to achieve the goal stated above. These actions should be distinct yet similar to the initial ones, maintaining strong relevance and consistency within the same domain (e.g., if deciding on item to select, suggest othern item options). Each action must be in the form of a subject and predicate. Ensure these additional actions are directly comparable to the initial ones. Provide only what to do, without explanations, in the following JSON format:

```json
{{
    "new_actions": ["Make action a", "Make action b", ...]
}}
```"""

    def __init__(self):
        self.web_retriever = WebRetriever().create()

    def create(self):
        llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.0)
        return (
            RunnablePassthrough.assign(
                retrieved_data=lambda state: self._retrieve_data(state)
            )
            | PromptTemplate.from_template(self.ACTION_EXPANSION_PROMPT_TEMPLATE)
            | llm
            | JsonOutputParser()
        )

    def _retrieve_data(self, state):
        web_search_query = f"User request was \"{state["user_request"]}\", so initially proposed actions are {state["initial_actions"]}. Please suggest proposed actions alternetives."
        return self.web_retriever.invoke({"input": web_search_query})
