import os
from googleapiclient.discovery import build
from pydantic import BaseModel

from ..chains.analysis_chain import AnalysisChain


class InitialAgent:

    def __init__(self):
        self.analysis_chain = AnalysisChain().create()
        self.pages_per_query = min(int(os.environ.get("PAGES_PER_QUERY", 3)), 10)  # Capped at 10
        self.api_key = os.environ["GOOGLE_API_KEY"]  # Required
        self.cse_id = os.environ["GOOGLE_CSE_ID"]    # Required
        self.google_service = build("customsearch", "v1", developerKey=self.api_key)

    def invoke(self, state):
        state = state | self.analysis_chain.invoke(state)

        query_uris = {}
        for query in state["queries"]:
            lang = state.get("language", "en")
            lang_param = f"lang_{lang}" if len(lang) == 2 else lang

            response = self.google_service.cse().list(
                q=query,
                cx=self.cse_id,
                num=self.pages_per_query,
                lr=lang_param
            ).execute()

            uris = [item["link"] for item in response.get("items", [])]
            query_uris[query] = uris

        return state | {
            "query_uris": query_uris
        }