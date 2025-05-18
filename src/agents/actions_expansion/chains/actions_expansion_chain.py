from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
import os


class ActionsExpansionChain:
    ACTION_EXPANSION_PROMPT_TEMPLATE = """Goal Definition:
{goal_definition}

Initial actions:
{initial_actions}

Retrieved data:
<RETRIEVED_DATA>
{retrieved_data}
</RETRIEVED_DATA>

Analyze the retrieved data to identify up to """+ os.getenv("EXPANDED_ACTIONS_MAX_LIMIT") + """ of the most reliable additional actions that can be taken to achieve the goal stated above. These actions should be distinct yet similar to the initial ones, maintaining strong relevance and consistency within the same domain (e.g., if deciding on item to select, suggest othern item options). Each action must be in the form of a subject and predicate. Ensure these additional actions are directly comparable to the initial ones. Provide only what to do, without explanations, in the following JSON format:

```json
{{
    "new_actions": ["Make action a", "Make action b", ...]
}}
```"""

    def create(self):
        llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.0)
        return (
            PromptTemplate.from_template(self.ACTION_EXPANSION_PROMPT_TEMPLATE)
            | llm
            | JsonOutputParser()
        )
