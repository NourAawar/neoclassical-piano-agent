import os
import re
import sqlite3
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import wikipedia 
import urllib.parse


#configuration 
CHROMA_DIR = "./chroma_db" 
DB_PATH = "./piano_db"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5"
MAX_ITER = 6 
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


#these resources are loaded once at startup 
embeddings = OllamaEmbeddings(model = EMBED_MODEL, base_url = OLLAMA_HOST)
#vectorstore = Chroma(persist_directory = CHROMA_DIR, embedding_function = embeddings)
vectorstore = None


#database setup (SQLite)
def init_db() -> None: 
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    #composers table
    cur.execute("""
                CREATE TABLE IF NOT EXISTS composers(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    name TEXT NOT NULL UNIQUE, 
                    nationality TEXT, 
                    born INTEGER, 
                    style TEXT)
                """)
    
    #pieces table
    cur.execute("""
                CREATE TABLE IF NOT EXISTS pieces(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    title TEXT NOT NULL, 
                    composer TEXT NOT NULL, 
                    album TEXT, 
                    mood TEXT, 
                    difficulty TEXT CHECK(difficulty in ('beginner', 'intermediate', 'advanced')))
                """)
    
    #listening history table 
    cur.execute("""
                CREATE TABLE IF NOT EXISTS listening_history(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    query TEXT NOT NULL, 
                    topic TEXT, 
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
                """)
    
    #here we seed composers only if the table is empty 
    cur.execute("SELECT COUNT(*) FROM composers")
    if cur.fetchone()[0] == 0: 
        composers_seed = [
            ("Ludovico Einaudi", "Italian", 1955, "Classical, New Age"), 
            ("Gibran Alcocer", "Mexican", 2003, "Ambient Solo Piano, New Age"), 
            ("Yann Tiersen", "French", 1970, "Cinematic neoclassical"), 
            ("Virginio Aiello", "Italian", 1975, "Neoclassical"), 
            ("Max Richter", "British", 1966, "Neoclassical"), 
        ]
        cur.executemany(
            "INSERT INTO composers (name, nationality, born, style) VALUES (?, ?, ?, ?)", 
            composers_seed
        )
    
    #here we seed pieces only if the table is empty 
    cur.execute("SELECT COUNT(*) FROM pieces")
    if cur.fetchone()[0] == 0: 
        pieces_seed = [
            ("Nuvole Bianche", "Ludovico Einaudi", "Una Mattina", "melancholic, hopeful", "intermediate"), 
            ("Experience", "Ludovico Einaudi", "In a Time Lapse", "cinematic, emotional", "advanced"), 
            ("Una Mattina", "Ludovico Einaudi", "Una Mattina", "gentle, reflective", "beginner"), 
            ("Idea 1", "Gibran Alcocer", "Ideas Collection", "melancholic, nostalgic", "intermediate"), 
            ("Idea 10", "Gibran Alcocer", "Ideas Collection", "melancholic, reflective", "intermediate"), 
            ("Idea 15", "Gibran Alcocer", "Ideas Collection", "melancholic, calm", "beginner"), 
            ("Idea 20", "Gibran Alcocer", "Ideas Collection", "melancholic, atmospheric", "intermediate"), 
            ("Idea 22", "Gibran Alcocer", "Ideas Collection", "calm, reflective", "beginner"), 
            ("Idea 25", "Gibran Alcocer", "Ideas Collection", "melancholic, emotional", "intermediate"), 
            ("Comptine d'un autre été", "Yann Tiersen", "Le Fabuleux Destin d'Amélie Poulain", "playful, wistful", "beginner"), 
            ("La valse d'Amélie", "Yann Tiersen", "Le Fabuleux Destin d'Amélie Poulain", "charming, lively", "intermediate"), 
            ("Winter Snow", "Virginio Aiello", "Single", "calm, nostalgic", "intermediate"), 
            ("Van Gogh", "Virginio Aiello", "Single", "emotional, melancholic", "beginner"), 
            ("On the Nature of Daylight", "Max Richter", "The Blue Notebooks", "sorrowful, expansive", "intermediate"), 
            ("Recomposed: Spring 1", "Max Richter", "Recomposed", "vibrant, textured", "advanced"), 
        ]
        cur.executemany(
            "INSERT INTO pieces (title, composer, album, mood, difficulty) VALUES (?, ?, ?, ?, ?)", 
            pieces_seed
        )
    
    conn.commit()
    conn.close()
    print("Database built.")

#changed the piano_data.txt so we have to build the chroma index again 
def init_vector_store() -> None: 
    global vectorstore 

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR): 
        print("Vector store already exists, loading...")
        vectorstore = Chroma(persist_directory = CHROMA_DIR, embedding_function = embeddings)
        return 
        
    print("Building vector store from new piano_data.txt...")

    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = TextLoader("piano_data.txt", encoding = "utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500, 
        chunk_overlap = 75, 
        separators = ["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    Chroma.from_documents(
        documents = chunks, 
        embedding = embeddings, 
        persist_directory = CHROMA_DIR
    )

    print("Vector store built and saved.")
    vectorstore = Chroma(persist_directory = CHROMA_DIR, embedding_function = embeddings)
    
    

def _search_rag(query: str) -> str: 
    docs = vectorstore.as_retriever(search_kwargs = {"k": 4}).invoke(query)
    if not docs: 
        return "No relevant information found in the knowledge base."
    
    return "\n\n".join([doc.page_content.strip() for doc in docs])

def _run_sql(query: str) -> str: 
    try: 
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(query)

        #for the SELECT queries
        if query.strip().upper().startswith("SELECT"): 
            rows = cur.fetchall()
            headers = [desc[0] for desc in cur.description] if cur.description else []
            conn.close()

            if not rows: 
                return "Query returned no results."
            
            lines = [" | ".join(str(v) for v in row) for row in rows]
            
            return " | ".join(headers) + "\n" + "\n".join(lines)

        #for INSERT or UPDATE or CREATE 
        conn.commit()
        affected = cur.rowcount
        conn.close()

        return f"Query executed successfully. The rows affected: {affected}."
    
    except Exception as e: 
        return f"SQL error: {str(e)}"

def _lookup_wikipedia(query: str) -> str: 
    try: 
        summary = wikipedia.summary(query, sentences = 5, auto_suggest = True)

        return summary 

    except wikipedia.exceptions.DisambiguationError as e: 
        try: 
            return wikipedia.summary(e.options[0], sentences = 5)
        except Exception: 
            return f"Wikipedia returned multiple matches for '{query}'. Try a more specific search."
    
    except Exception as e: 
        return f"Wikipedia look up failed: {str(e)}"

def _find_youtube(query: str) -> str: 
    search_query = urllib.parse.quote(query + " piano")
    url = f"https://www.youtube.com/results?search_query={search_query}"

    return (
        f"YouTube search for '{query}':\n"
        f"{url}\n"
        f"Click the link to browse performances and recordings."
    )


#the llm instances 
llm = ChatOllama(model = CHAT_MODEL, temperature = 0, base_url = OLLAMA_HOST)
presenter_llm = ChatOllama(model = CHAT_MODEL, temperature = 0.7, base_url = OLLAMA_HOST)


#the state 
class AgentState(TypedDict): 
    messages: Annotated[list, add_messages]
    next_agent: str 
    iteration_count: int 


#the guardrail nodes 
INJECTION_PATTERNS  = [
    "ignore your instructions", 
    "ignore previous", 
    "disregard your", 
    "you are now", 
    "system prompt:", 
    "forget everything", 
    "new persona", 
    "act as if",
]

#some personal identifiable information (PII) patterns 
PII_PATTERNS = [
    r'\b\d{3}-\d{2}-\d{4}\b', 
    r'\b\d{16}\b', 
    r'password\s*[:=]\s*\S+', 
    r'credit.?card\s*[:=]\s*\S+', 
]

def input_guard(state: AgentState) -> dict: 
    last_msg = state["messages"][-1].content.lower()

    for pattern in INJECTION_PATTERNS: 
        if pattern in last_msg: 
            return {
                "messages": [AIMessage(content = (
                    "I detected a potentially unsafe request pattern."
                    "Please rephrase your question about piano music."
                ))], 
                "next_agent": "BLOCKED", 
                "iteration_count": 0, 
            }
    
    return {"next_agent": "supervisor", "iteration_count": 0}

def output_guard(state: AgentState) -> dict: 
    last_msg = state["messages"][-1].content

    for pattern in PII_PATTERNS: 
        if re.search(pattern, last_msg, re.IGNORECASE): 
            return {
                "messages": [AIMessage(content = (
                    "The response contained potentially sensitive information and was blocked. Please get support if you need help."
                ))]
            }
    
    return {}


#supervisor node 
SPECIALIST_OPTIONS = ["rag_agent", "sql_agent", "wiki_agent", "youtube_agent", "FINISH"]

SUPERVISOR_SYSTEM = f"""You are a supervisor managing specialist agents for a piano music assistant. 
Decide which specialist should act next, or FINISH if you have enough information.

Specialists: 
    rag_agent - emotional descriptions, learning advice, piece atmosphere composer style. 
    sql_agent - structured database queries: list pieces, filter by difficulty/mood, log history
    wiki_agent - biographical facts about a composer: birthplace, education, awards, career history
    youtube_agent - only when user explicitly asks for a video or link to watch/listen

Rules: 
    - Always try rag_agent first for music-feel or learning questions
    - Use sql_agent for ANY question about a specific piece title (e.g. "who made X", "who is the artist of X", "what album is X from") - piece data lives in the database
    - Use sql_agent for structured queries ("show beginner pieces", "list composers", "save to history") 
    - Use wiki_agent ONLY for biolographical facts about a composer - NOT for piece lookups
    - Use youtube_agent only when the user explicitly wants a link or video
    - Respond with ONLY one word from: {SPECIALIST_OPTIONS}
    - Once the specialists have gathered enough information, respond with FINISH
"""

def supervisor (state: AgentState) -> dict: 
    count = state.get("iteration_count", 0) + 1

    if count > MAX_ITER: 
        return {
            "messages": [AIMessage(content = (
                "I have reached the maximum number of steps for this request. "
                "Here is what I found so far; feel free to ask a follow-up."
            ))], 
            "next_agent": "FINISH", 
            "iteration_count": count, 
        }
    
    last = state["messages"][-1]
    last_content = last.content.strip() if last.content else ""
    is_routing_word = last_content in SPECIALIST_OPTIONS
    is_short_label = len(last_content.split()) <= 2
    is_real_response = (
        last.type == "ai"
        and last_content
        and not is_routing_word
        and not is_short_label 
        and not last_content.startswith("[")
    )
    if is_real_response: 
        return {
            "messages": [],
            "next_agent": "FINISH", 
            "iteration_count": count, 
        }
    
    messages = [SystemMessage(content = SUPERVISOR_SYSTEM)] + state["messages"]
    response = llm.invoke(messages)
    decision = response.content.strip()

    if decision not in SPECIALIST_OPTIONS: 
        decision = "FINISH"
    
    return {
        "messages": [response], 
        "next_agent": decision, 
        "iteration_count": count, 
    }

def route_from_supervisor(state: AgentState) -> str: 
    return state["next_agent"]


def _get_human_query(state: AgentState) -> str: 
    for msg in reversed(state["messages"]): 
        if msg.type == "human": 
            return msg.content
    return "" 


#rag_node 
def rag_node(state: AgentState) -> dict: 
    user_query = _get_human_query(state)
    context = _search_rag(user_query)
    response = presenter_llm.invoke([
        SystemMessage(content = (
            "You are a passionate specialist in modern neoclassical piano music. "
            "Answer the user's question using ONLY the context provided below. "
            "Speak in terms of feeling, atmosphere, and emotion. "
            "Never use note names like C, D, E; use solfege if needed. "
            "If the context does not contain the answer, say so honestly."
        )), 
        HumanMessage(content = f"Context:\n{context}\n\nQuestion: {user_query}")
    ])

    return {"messages": [response]}

#sql_node 
def sql_node(state: AgentState) -> dict:
    user_query = _get_human_query(state)

    sql_response = llm.invoke([
        SystemMessage(content = (
            "You are a SQL expert. Write a single valid SQLite query for the user's question.\n"
            "Tables:\n"
            "composers (id, name, nationality, born, style)\n"
            "pieces    (id, title, composer, album, mood, difficulty)\n"
            "listening_history (id, query, topic, timestamp)\n\n"
            "Rules:\n"
            "- difficulty values are exactly: beginner, intermediate, advanced\n"
            "- For INSERT into listening_history: INSERT INTO listening_history (query, topic) VALUES ('...','...')\n"
            "- Respond with ONLY the raw SQL. No explanation, no markdown, no backticks."
        )),
        HumanMessage(content = user_query)
    ])
    generated_sql = sql_response.content.strip().replace("```sql","").replace("```","").strip()

    db_result = _run_sql(generated_sql)

    response = presenter_llm.invoke([
        SystemMessage(content = (
            "You are presenting database results to a user. "
            "You MUST use ONLY the exact data from the database result below. "
            "Do NOT add, substitute, or correct any names from your own knowledge. "
            "The data is correct as-is. Present it warmly and readably."
        )),
        HumanMessage(content = (
            f"User question: {user_query}\n\n"
            f"Exact database result:\n{db_result}\n\n" 
            "Present ONLY this data faithfully." 
        ))
    ])
    return {"messages": [response]}

def wiki_node(state: AgentState) -> dict:
    user_query = _get_human_query(state)

    term_response = llm.invoke([ 
        SystemMessage(content = ( 
            "Extract the composer or person name to search on Wikipedia. " 
            "Respond with ONLY the name, nothing else." 
        )),
        HumanMessage(content = user_query)
    ])
    search_term = term_response.content.strip()
    wiki_result = _lookup_wikipedia(search_term) 

    response = presenter_llm.invoke([
        SystemMessage(content = ( 
            "You are a biographical research specialist. " 
            "Answer the user's question using ONLY the Wikipedia extract provided. " 
            "Be warm and informative."
        )),
        HumanMessage(content = f"Wikipedia result:\n{wiki_result}\n\nQuestion: {user_query}")
    ])
    return {"messages": [response]}

def youtube_node(state: AgentState) -> dict:
    user_query = _get_human_query(state)

    term_response = llm.invoke([
        SystemMessage(content = (
            "Extract the piece or composer name to search on YouTube. "
            "Respond with ONLY the search term, nothing else."
        )), 
        HumanMessage(content = user_query)
    ])
    search_term = term_response.content.strip() 
    yt_result = _find_youtube(search_term)

    response = presenter_llm.invoke([
        SystemMessage(content = (
            "You are a music discovery assistant. "
            "Share the YouTube search link provided below with the user in a warm, friendly way. "
            "You MUST copy the exact URL from the link provided."
            "NEVER write a youtube.com/watch URL. "
            "NEVER invent or modify a URL. "
            "Only use the search link you are given. "
            "NEVER return 'https://www.youtube.com/watch?v=example'.")),
        HumanMessage(content = f"YouTube link:\n{yt_result}\n\nUser asked: {user_query}")
    ])
    return {"messages": [response]}


#the input guard router
def input_guard_router(state: AgentState) -> Literal["supervisor", "end"]: 
    if state.get("next_agent") in ("FINISH", "BLOCKED"): 
        return "end"
    
    return "supervisor"


#now we build the graph 
def build_graph() -> object: 
    graph = StateGraph(AgentState)

    #guardrail nodes 
    graph.add_node("input_guard", input_guard)
    graph.add_node("output_guard", output_guard)

    #the core nodes
    graph.add_node("supervisor", supervisor)
    graph.add_node("rag_agent", rag_node)
    graph.add_node("sql_agent", sql_node)
    graph.add_node("wiki_agent", wiki_node)
    graph.add_node("youtube_agent", youtube_node)

    #the edges 
    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges(
        "input_guard", 
        input_guard_router, 
        {"supervisor": "supervisor", "end": END}
    )

    graph.add_conditional_edges(
        "supervisor", 
        route_from_supervisor, 
        {
            "rag_agent": "rag_agent", 
            "sql_agent": "sql_agent", 
            "wiki_agent": "wiki_agent", 
            "youtube_agent": "youtube_agent", 
            "FINISH": "output_guard"
        }
    )

    for agent in ["rag_agent", "sql_agent", "wiki_agent", "youtube_agent"]: 
        graph.add_edge(agent, "supervisor")
    
    graph.add_edge("output_guard", END)

    return graph.compile()


#now we build the public api 
agent_graph = build_graph() 

def run_agent(user_input: str, history: list) -> tuple[str, list]:  
    history = history + [HumanMessage(content = user_input)]
    
    result = agent_graph.invoke({
        "messages": history, 
        "next_agent": "", 
        "iteration_count": 0, 
    })

    response = result["messages"][-1].content
    updated_history = result["messages"]

    return response, updated_history 

def main() -> None: 
    print("Piano Multi-Agent System")
    print("Agents: RAG | SQL | Wikipedia | YouTube")
    print("Guardrails: input injection | output PII | iteration cap")
    print("\nAsk me anything about modern neoclassical piano music.")
    print("Type 'quit' or 'exit' to end the conversation. \n")

    init_db()

    init_vector_store()

    history: list = []

    while True: 
        try: 
            user_input = input("You: ").strip()
        
        except (KeyboardInterrupt, EOFError): 
            print("\nGoodbye! Keep listening to amazing music.")
            break 

        if not user_input: 
            continue 

        if user_input.lower() in ("quit", "exit"): 
            print("Goodbye! It was great talking neoclassical piano music with you!")
            break
        
        try: 
            response, history = run_agent(user_input, history)
            print(f"\nAgent: {response}\n")
        
        except Exception as e: 
            print(f"\n[Error: {e}]")
            print("Make sure 'ollama serve' is running in another terminal.\n")

if __name__ == "__main__": 
    main()
    