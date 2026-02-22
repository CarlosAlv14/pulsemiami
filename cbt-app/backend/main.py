from fastapi import FastAPI
from dotenv import load_dotenv
from database import supabase
from session_graph import session_graph
import os

load_dotenv()

app = FastAPI(title="CBT Therapy Aid API")

@app.get("/")
def root():
    return {"status": "CBT backend running"}

@app.get("/health")
def health():
    return {
        "anthropic_key_loaded": bool(os.getenv("ANTHROPIC_API_KEY")),
        "supabase_url_loaded": bool(os.getenv("SUPABASE_URL")),
        "supabase_key_loaded": bool(os.getenv("SUPABASE_KEY"))
    }

@app.get("/test-db")
def test_db():
    result = supabase.table("therapists").select("*").execute()
    return {"connected": True, "rows": result.data}

@app.post("/session/run")
def run_session(payload: dict):
    result = session_graph.invoke({
        "client_id": payload.get("client_id", "test-client"),
        "session_plan_id": payload.get("session_plan_id", "test-plan"),
        "raw_content": payload.get("raw_content", ""),
        "current_step": "start",
        "tolerance": payload.get("tolerance", {
            "reading_complexity": 50,
            "vocabulary_range": 50,
            "abstraction_comfort": 50,
            "working_memory_capacity": 50,
            "frustration_sensitivity": 50
        }),
        "interaction_log": [],
        "adapted_content": "",
        "client_response": payload.get("client_response", ""),
        "report": ""
    })
    return result