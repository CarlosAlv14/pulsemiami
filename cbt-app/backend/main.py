from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
from database import supabase
from session_graph import session_graph
import whisper
import tempfile
import os
import re

load_dotenv()

app = FastAPI(title="CBT Therapy Aid API")

whisper_model = whisper.load_model("base")

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

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    result = whisper_model.transcribe(tmp_path)
    os.unlink(tmp_path)

    return {"transcription": result["text"]}

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

    # Save to database
    supabase.table("session_runs").insert({
        "session_plan_id": result["session_plan_id"],
        "client_id": result["client_id"],
        "vector_snapshot": result["tolerance"],
        "report": result["report"]
    }).execute()

    print("\n--- THERAPIST REPORT ---")
    print(result["report"])
    print("------------------------\n")

    return result

@app.get("/session/latest/{client_id}")
def get_latest_session(client_id: str):
    result = supabase.table("session_runs") \
        .select("*") \
        .eq("client_id", client_id) \
        .order("started_at", desc=True) \
        .limit(1) \
        .execute()

    if not result.data:
        return {"message": "No sessions found for this client"}

    session = result.data[0]

    # Strip markdown formatting
    report = session["report"]
    report = re.sub(r'#{1,6}\s*', '', report)
    report = re.sub(r'\*\*(.*?)\*\*', r'\1', report)
    report = re.sub(r'\*(.*?)\*', r'\1', report)
    report = re.sub(r'\n{3,}', '\n\n', report)
    report = report.strip()

    return {
        "client_id": client_id,
        "report": report,
        "tolerance": session["vector_snapshot"],
        "session_id": session["id"],
        "date": session["started_at"]
    }
