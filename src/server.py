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
        "is_valid": False
    }
    return graph.invoke(initial_state)


if __name__ == "__main__":
    uvicorn.run(app)
