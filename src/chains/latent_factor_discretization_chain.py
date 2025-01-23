from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
import os


class LatentFactorDiscretizationChain:
    STATES_DISCRETIZATION_PROMPT_TEMPLATE = """Considering the following latent factor:
{latent_factor}

In the context of the decision with goal defined below:
{goal_definition}

Propose up to """ + os.getenv("LATENT_FACTOR_VALUES_MAX_LIMIT") + """ possible future outcomes that this latent factor can take. These values should represent different plausible scenarios such as "the event will happen", "the event will not happen", "the event will happen partially", etc. Be concise.

Respond in the following JSON format:
{{
    "latent_factor_values": ["Possible Outcome 1", "Possible Outcome 2", "..."]
}}
"""

    def create(self):
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
        return (
            PromptTemplate.from_template(self.STATES_DISCRETIZATION_PROMPT_TEMPLATE)
            | self.llm
            | JsonOutputParser()
        )
