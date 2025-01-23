from chains.latent_factors_identification_chain import LatentFactorsIdentificationChain
from chains.latent_factor_discretization_chain import LatentFactorDiscretizationChain
from retrievers.web_retriever import WebRetriever



class StatesEnumerationAgent:

    def __init__(self):
        self.latent_factors_identification_chain = LatentFactorsIdentificationChain().create()
        self.latent_factor_discretization_chain = LatentFactorDiscretizationChain().create()
        self.web_retriever = WebRetriever().create()


    def invoke(self, state):
        web_search_query = f"The user is making a decision with the following goal: {state['goal_definition']}. Please explore relevant current online data, focusing on information pertinent to any potential latent risk factors that can affect the decision-making process and goal achievement."
        state = state | {"input": web_search_query}
        state = state | {"retrieved_data": self.web_retriever.invoke({"input": web_search_query})}
        #print(state["retrieved_data"])

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
