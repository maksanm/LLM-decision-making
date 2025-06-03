from .chains.latent_factors_identification_chain import LatentFactorsIdentificationChain
from .chains.latent_factor_discretization_chain import LatentFactorDiscretizationChain


class StatesEnumerationAgent:

    def __init__(self):
        self.latent_factors_identification_chain = LatentFactorsIdentificationChain().create()
        self.latent_factor_discretization_chain = LatentFactorDiscretizationChain().create()

    def invoke(self, state):
        try:
            identification_result = self.latent_factors_identification_chain.invoke(state)

            states_batch = []
            for factor_name in identification_result["latent_factors"]:
                states_batch.append(state | {"latent_factor": factor_name})

            discretization_results = self.latent_factor_discretization_chain.batch(states_batch)

            state_space = {}
            for i, result in enumerate(discretization_results):
                state_space = state_space | {identification_result["latent_factors"][i]: result["latent_factor_values"]}
            return {"state_space": state_space}
        except Exception as e:
            print(f"Error parsing the enumerated states: {e}")
            raise e
