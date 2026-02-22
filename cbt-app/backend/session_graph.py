from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from typing import TypedDict
import os

llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=os.getenv("ANTHROPIC_API_KEY"))

# Session state — everything the graph knows at any point
class SessionState(TypedDict):
    client_id: str
    session_plan_id: str
    raw_content: str
    current_step: str
    tolerance: dict
    interaction_log: list
    adapted_content: str
    client_response: str
    report: str

# Node 1 — adapt the therapist's content to the client's current tolerance
def adapt_content(state: SessionState) -> SessionState:
    tolerance = state["tolerance"]
    raw = state["raw_content"]

    prompt = f"""You are helping deliver CBT therapy to a client with intellectual disabilities.

Therapist's raw content:
{raw}

Client's current tolerance levels (0-100):
- Reading complexity: {tolerance.get("reading_complexity", 50)}
- Vocabulary range: {tolerance.get("vocabulary_range", 50)}
- Abstraction comfort: {tolerance.get("abstraction_comfort", 50)}
- Working memory capacity: {tolerance.get("working_memory_capacity", 50)}

Rewrite the content so it is appropriate for this client. Use simple, concrete, everyday language. 
Avoid jargon. Use short sentences. Be warm and encouraging. Do not infantilize."""

    response = llm.invoke(prompt)
    state["adapted_content"] = response.content
    state["current_step"] = "concept_introduction"
    return state

# Node 2 — evaluate the client's response and update tolerance vector
def evaluate_response(state: SessionState) -> SessionState:
    tolerance = state["tolerance"]
    client_response = state["client_response"]
    adapted_content = state["adapted_content"]

    prompt = f"""You are assessing a therapy session interaction.

What was delivered to the client:
{adapted_content}

Client's response:
{client_response}

Based on this interaction, suggest small adjustments (-5 to +5) to these tolerance scores:
- reading_complexity
- vocabulary_range  
- abstraction_comfort
- working_memory_capacity
- frustration_sensitivity

Respond in this exact format:
reading_complexity: <number>
vocabulary_range: <number>
abstraction_comfort: <number>
working_memory_capacity: <number>
frustration_sensitivity: <number>"""

    response = llm.invoke(prompt)
    
    # Parse adjustments and update tolerance
    for line in response.content.strip().split("\n"):
        if ":" in line:
            key, val = line.split(":")
            key = key.strip()
            try:
                adjustment = float(val.strip())
                if key in tolerance:
                    tolerance[key] = max(0, min(100, tolerance[key] + adjustment))
            except:
                pass

    state["tolerance"] = tolerance
    state["current_step"] = "generate_report"
    return state

# Node 3 — generate therapist report
def generate_report(state: SessionState) -> SessionState:
    prompt = f"""You are summarizing a CBT therapy session for the supervising therapist.

Session content delivered:
{state["adapted_content"]}

Client response:
{state["client_response"]}

Updated tolerance profile:
{state["tolerance"]}

Write a brief, clinical summary for the therapist. Note how the client engaged, any signs of difficulty or success, and one recommendation for the next session. Use professional language."""

    response = llm.invoke(prompt)
    state["report"] = response.content
    state["current_step"] = "complete"
    return state

# Build the graph
def build_session_graph():
    graph = StateGraph(SessionState)
    
    graph.add_node("adapt_content", adapt_content)
    graph.add_node("evaluate_response", evaluate_response)
    graph.add_node("generate_report", generate_report)
    
    graph.set_entry_point("adapt_content")
    graph.add_edge("adapt_content", "evaluate_response")
    graph.add_edge("evaluate_response", "generate_report")
    graph.add_edge("generate_report", END)
    
    return graph.compile()

session_graph = build_session_graph()