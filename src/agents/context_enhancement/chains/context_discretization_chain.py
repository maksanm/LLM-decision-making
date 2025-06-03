from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI


class ContextDiscretizationChain:
    CONTEXT_DISCRETIZATION_PROMPT_TEMPLATE = """
You are given initial context in free text format and states space, which is a dict, each key is a latent factor, each value is a list of possible value phrasings for that factor.

Task:
For each latent factor (keep their order), extract all fragments from initial context relevant to (a) the factor as a whole, or (b) any of its possible values. For factors with no relevant info in the context, return an empty string.

**Output:**  
Return a JSON:
```json
{"contexts": [context_for_factor1, context_for_factor2, ...]}
```
—list order as in `states_space` keys, each entry a string (possibly empty).

Notes:
- Use only the content from `initial_context` (no inventions or summaries).
- Be as precise/strict as possible; omit irrelevant info.
- Variables:  
  - `states_space` = {states_space}  
  - `initial_context` = {initial_context}

"""

    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0.0)

    def create(self):
        return (
            PromptTemplate.from_template(self.CONTEXT_DISCRETIZATION_PROMPT_TEMPLATE)
            | self.llm
            | JsonOutputParser()
        )
