from chains.input_validation_chain import InputValidationChain
from chains.prompt_decomposition_chain import PromptDecompositionChain


class InitialAnalysisAgent:

    def __init__(self):
        self.validation_chain = InputValidationChain().create()
        self.prompt_decomposition_chain = PromptDecompositionChain().create()


    def invoke(self, state):
        is_valid_str = self.validation_chain.invoke(state)
        is_valid = self._parse_bool(is_valid_str)
        if not is_valid:
            return {
                "is_valid": False
            }

        try:
            decomposed_prompt = self.prompt_decomposition_chain.invoke(state)
            return {
                "is_valid": True,
                "goal_definition": decomposed_prompt["goal_definition"],
                "initial_context": decomposed_prompt["initial_context"],
                "action_space": decomposed_prompt["action_space"]
            }
        except Exception as e:
            print(f"Error parsing the decomposed prompt: {e}")
            raise e


    def _parse_bool(self, str):
        if str == "True": return True
        elif str == "False": return True
        raise Exception("Unable to parse the LLM output")

