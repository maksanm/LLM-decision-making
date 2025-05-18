from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
import os


class LatentFactorDiscretizationChain:
    STATES_DISCRETIZATION_PROMPT_TEMPLATE = """Considering the following latent factor:
{latent_factor}

In the context of the decision with the goal defined below:
{goal_definition}

Propose up to """ + os.getenv("LATENT_FACTOR_VALUES_MAX_LIMIT") + """ possible future outcomes that this latent factor can take. These values should represent different plausible scenarios, such as "the event will happen," "the event will not happen," "the event will happen partially," etc. Ensure that each outcome is concise and limited to a brief statement, avoiding any descriptions of what the outcome results in or how it influences the decision.

Respond in the following JSON format:

```json
{{
    "latent_factor_values": ["Concise Outcome 1", "Concise Outcome 2", "..."]
}}
```"""

    def create(self):
        self.llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0.0)
        return (
            PromptTemplate.from_template(self.STATES_DISCRETIZATION_PROMPT_TEMPLATE)
            | self.llm
            | JsonOutputParser()
        )
