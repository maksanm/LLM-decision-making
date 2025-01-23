from dotenv import load_dotenv

load_dotenv()

from graph import DecisionGraph

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = DecisionGraph().create()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


@app.post("/debug")
async def debug(user_request: str):
    initial_state = {
        "user_request": user_request,
        "goal_definition": "",
        "initial_context": "",
        "action_space": [],
        "is_valid": False,
        "expanded_actions": "",
        "state_space": {}
    }
    return graph.invoke(initial_state)


if __name__ == "__main__":
    uvicorn.run(app)


'''
from langchain_core.runnables.graph import MermaidDrawMethod
image_data = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
file_path = "docs/graph.png"
with open(file_path, "wb") as file:
    file.write(image_data)
'''
