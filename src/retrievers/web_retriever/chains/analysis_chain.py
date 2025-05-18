import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI


class AnalysisChain:
    REPHRASE_PROMPT_TEMPLATE = "Task : Generate up to " + os.getenv("QUERIES_MAX_LIMIT") + """ Google search queries that best help the user reach the goal implied in the input description and reveal any latent risks. Use the specified language or infer it. Return clean JSON only.

Rules
• Extract the main intent, place, time, risk cues, etc.
• Use clear, real-world search phrasing with synonyms where helpful; no repeats.
• 1 - """ + os.getenv("QUERIES_MAX_LIMIT") + """ unique queries, all in the chosen language.
• No extra keys or commentary.

Example
Input
{{
  "input": "The user is making a decision with the following goal: 'Wybierz odzież dla spaceru w Warszawie na jutro'. Please explore relevant current online data, focusing on information pertinent to any potential latent risk factors that can affect the decision-making process and goal achievement."
}}

Output
{{
  "language": "pl",
  "queries": [
    "prognoza pogody Warszawa jutro",
    "zalecenia odzieżowe na spacer Warszawa",
    "ryzyko opadów Warszawa jutro",
    "alerty pogodowe Warszawa 24h",
    "temperatura odczuwalna Warszawa jutro"
  ]
}}

Input:
{{
  "query": "{input}",
}}

Output:
"""

    def create(self):
        llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0.0)
        return (
            PromptTemplate.from_template(self.REPHRASE_PROMPT_TEMPLATE)
            | llm
            | JsonOutputParser()
        )
