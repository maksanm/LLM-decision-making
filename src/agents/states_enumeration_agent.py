from chains.states_identification_chain import StatesIdentificationChain
from chains.states_discretization_chain import StatesDiscretizationChain



class StatesEnumerationAgent:

    def __init__(self):
        self.states_enumeration_chain = StatesIdentificationChain().create()
        self.states_discretization_chain = StatesDiscretizationChain().create()


    def invoke(self, state):
        search_query = f"Defined Goal: {state['goal_definition']}. Please explore relevant current online data, focusing on information pertinent to the defined goal and identifying any potential risk factors that can affect the decision-making process"
        state = state | {"search_query": search_query}

        try:
            #response = self.states_enumeration_chain.invoke(state)
            return {
            }
        except Exception as e:
            print(f"Error parsing the enumerated states: {e}")
            raise e
