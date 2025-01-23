from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI


class PromptDecompositionChain:
    DECOMPOSITION_PROMPT_TEMPLATE = """Extract the following information from the user request:
1. Goal definition
2. Initial context, independent of the goal definition, containing user-provided decision-supportive data; if none is provided, return an empty string
3. Action space comprises concrete actions clearly articulated in the user request in the form of a subject and predicate. Each action must be directly applicable and executable, leaving no room for vague or abstract considerations. Do not invent new actions. For example, if the user is considering using a car or bicycle, and other vehicles are also being considered, return only 'Choose car' and 'Choose bicycle' as options.

Respond in the following JSON format:
{{
    "goal_definition": "Goal definition X",
    "initial_context": "Initial context X",
    "action_space": [["Action a", "Action b", ...]
}}

<EXAMPLE>
User request:
I'm a farmer and deciding what culture to plant. I can plant pears or avocados. Maybe I should consider another crop? The next year the demand for avocados is going to increase, the demand for pears would stay the same.

Response:
{{
    "goal_definition": "Decide which crop to plant.",
    "initial_context": "The demand for avocados is expected to increase next year, while the demand for pears will remain constant.",
    "action_space": ["Plant avocados", "Plant pears"]
}}
</EXAMPLE>

User request:
{user_request}

Response:
"""

    def create(self):
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
        return (
            PromptTemplate.from_template(self.DECOMPOSITION_PROMPT_TEMPLATE)
            | self.llm
            | JsonOutputParser()
        )
