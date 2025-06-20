# mini_flow.py
from pathlib import Path
import joblib
from langgraph.graph import StateGraph, START, END

# ------------------------------------------------------------------
#  Load artefacts saved with train_and_eval.py
# ------------------------------------------------------------------
ROOT = Path(__file__).parent               # folder where this file lives
vectorizer = joblib.load(ROOT / "tfidf_vectorizer.pkl")
model      = joblib.load(ROOT / "logreg_model.pkl")

# ------------------------------------------------------------------
#  Node 1: classify the incoming e-mail
# ------------------------------------------------------------------
def classify_email(state: dict) -> dict:
    text   = state["body"]
    intent = model.predict(vectorizer.transform([text]))[0]
    return {"intent": intent}              # merged into graph state

# ------------------------------------------------------------------
#  Node 2: trigger the (fake) workflow side-effect
# ------------------------------------------------------------------
def trigger_workflow(state: dict) -> dict:
    print(f"⚡ Triggering {state['intent']} workflow")
    return state                           # nothing else to add

# ------------------------------------------------------------------
#  Build the two-node LangGraph
# ------------------------------------------------------------------
graph = StateGraph(dict)                   # graph state = plain dict
graph.add_node("ClassifyEmail",   classify_email)
graph.add_node("TriggerWorkflow", trigger_workflow)

graph.add_edge(START,            "ClassifyEmail")   # entrypoint
graph.add_edge("ClassifyEmail",  "TriggerWorkflow")
graph.add_edge("TriggerWorkflow", END)

flow = graph.compile()

# ------------------------------------------------------------------
#  Helper used by CLI / FastAPI
# ------------------------------------------------------------------
def run(text: str) -> dict:
    result = flow.invoke({"body": text})
    return {"intent": result["intent"], "status": "triggered"}

# ------------------------------------------------------------------
#  Allow quick terminal test:
#      python mini_flow.py "some email text"
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys, json
    print(json.dumps(run(sys.argv[1]), indent=2))
