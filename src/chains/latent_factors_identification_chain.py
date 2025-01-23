from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
import os

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

    def create(self):
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
        return (
            PromptTemplate.from_template(self.LATENT_FACTORS_IDENTIFICATION_PROMPT_TEMPLATE)
            | self.llm
            | JsonOutputParser()
        )
