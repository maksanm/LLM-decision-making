from ..chains.aggregation_chain import AggregationChain


class AggregationAgent:

    def __init__(self):
        self.aggregation_chain = AggregationChain().create()


    def invoke(self, state):
        result = self.aggregation_chain.invoke(state)

        import json
        with open(f"retrieve_debug_{state["input"][0:3]}.json", "a") as f:
            json.dump(state | result, f, indent=2)

        return result

