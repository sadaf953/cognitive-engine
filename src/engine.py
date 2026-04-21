import os
from typing import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

class BotPost(BaseModel):
    bot_id: str = Field(description="The unique ID of the bot")
    topic: str = Field(description="The topic of the research and post")
    post_content: str = Field(description="The final 280-character post content")

class GraphState(TypedDict):
    persona: str
    topic: str
    search_results: str
    final_post: dict

@tool
def mock_searxng_search(query: str):
    """Returns hardcoded, recent news headlines based on keywords."""
    q = query.lower()
    if "crypto" in q or "bitcoin" in q:
        return "Bitcoin hits new all-time high amid regulatory ETF approvals"
    
    if "ai" in q or "model" in q:
        return "OpenAI just released a new model that might replace junior developers."
    
    return "The Federal Reserve keeps interest rates steady; markets rally on ROI expectations."

llm = ChatGroq(temperature=0.7, model_name="llama-3.3-70b-versatile")


def decide_search_node(state: GraphState):
    """Node 1: Decide Search. The LLM decides the topic and search query."""
    print("--- NODE 1: DECIDE SEARCH ---")
    persona = state["persona"]
    prompt = f"You are {persona}. Decide what topic you want to post about today and format a search query. Return ONLY the search query text."
    response = llm.invoke(prompt)
    return {"topic": response.content}

def web_search_node(state: GraphState):
    """Node 2: Web Search. Executes the mock_searxng_search tool."""
    print("--- NODE 2: WEB SEARCH ---")
    query = state["topic"]
    results = mock_searxng_search.invoke(query)
    return {"search_results": results}

def draft_post_node(state: GraphState):
    """Node 3: Draft Post. Uses Persona + Search Context."""
    print("--- NODE 3: DRAFT POST ---")
    persona = state["persona"]
    context = state["search_results"]
    
    structured_llm = llm.with_structured_output(BotPost)
    
    prompt = f"""
    SYSTEM PROMPT: You are {persona}.
    CONTEXT: {context}.
    TOPIC: {state['topic']}.
    
    TASK: Generate a highly opinionated, 280-character post about the context. 
    Match your persona perfectly.
    """
    
    response_object = structured_llm.invoke(prompt)
    return {"final_post": response_object.model_dump()}


workflow = StateGraph(GraphState)


workflow.add_node("Decide Search", decide_search_node)
workflow.add_node("Web Search", web_search_node)
workflow.add_node("Draft Post", draft_post_node)

workflow.set_entry_point("Decide Search")
workflow.add_edge("Decide Search", "Web Search")
workflow.add_edge("Web Search", "Draft Post")
workflow.add_edge("Draft Post", END)

app = workflow.compile()