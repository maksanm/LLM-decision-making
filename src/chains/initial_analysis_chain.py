from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableBranch, RunnablePassthrough


class InitialAnalysisChain:
    VALIDATION_PROMPT_TEMPLATE = """Validate if the user request contains a clear goal that requires a decision. Return 'TRUE' if it does, otherwise return 'FALSE'.

User request:
{user_request}

Response:
"""

    DECOMPOSITION_PROMPT_TEMPLATE = """User request:
{user_request}

Extract the following information from the above request:
1. Goal definition
2. Initial context, independent of the goal definition, containing user-provided decision-supportive data; if none is provided, return an empty string
3. Action space comprises concrete actions clearly articulated in the user request. Each action must be directly applicable and executable, leaving no room for vague or abstract considerations. For example, if the user is considering using a car or bicycle, and other vehicles are also being considered, return only 'car' and 'bicycle' as options.

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
    "action_space": ["Avocados", "Pears"]
}}
</EXAMPLE>

Response:
"""

    def create(self):
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
        return (
            RunnablePassthrough.assign(
                is_valid=PromptTemplate.from_template(self.VALIDATION_PROMPT_TEMPLATE)
                    | llm
                    | StrOutputParser()
            )
            | RunnableBranch(
                (lambda state: state["is_valid"] == "FALSE", {}),
                lambda state: self._decompose_user_prompt(state, llm)
            )
        )

    def _decompose_user_prompt(self, state, llm):
        decomposition_chain = (
            PromptTemplate.from_template(self.DECOMPOSITION_PROMPT_TEMPLATE)
            | llm
            | JsonOutputParser()
        )
        try:
            decomposed_prompt = decomposition_chain.invoke(state)
            return {
                "goal_definition": decomposed_prompt["goal_definition"],
                "initial_context": decomposed_prompt["initial_context"],
                "action_space": decomposed_prompt["action_space"]
            }
        except Exception as e:
            print(f"Error parsing the decomposed prompt: {e}")
            raise e

