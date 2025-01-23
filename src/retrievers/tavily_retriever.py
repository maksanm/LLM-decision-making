from langchain_community.retrievers import TavilySearchAPIRetriever


class TavilyRetriever:

    def __init__(self, k=1, domains=None):
        self.retriever = TavilySearchAPIRetriever(
            k=k,
            include_generated_answer=True,
            include_domains=domains, # Use predifined data sources specific to the decision domain
            search_depth="advanced"
        )


    def invoke(self, search_query):
        return (self.retriever | self._format_tavily_response).invoke(search_query)


    def _format_tavily_response(self, retrieved_documents):
        formatted_response = []
        for document in retrieved_documents:
            content = document.page_content
            formatted_response.append(f"{content}\n")
        return "\n".join(formatted_response)
