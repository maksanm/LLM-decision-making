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

from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
image_data = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)

# Define the file path where you want to save the image
file_path = "graph.png"

# Write the image data to the file
with open(file_path, "wb") as file:
    file.write(image_data)



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
        "expanded_actions": ""
    }
    return graph.invoke(initial_state)


if __name__ == "__main__":
    uvicorn.run(app)
