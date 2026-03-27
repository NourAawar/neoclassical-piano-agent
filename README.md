# Neoclassical Piano Multi-Agent 

A multi-agent neoclassical piano music assistant built with LangGraph, RAG, SQLite, and Ollama. Dockerized with Dockerfile and Docker Compose. 

---

## What it does 

A conversational assistant for modern neoclassical piano music. It routes each question to the right specialist agent depending on what is being asked:

- **RAG agent** — answers questions about the emotional feel, atmosphere, and learning advice for pieces and composers, using a local Chroma vector store built from a curated knowledge base 
- **SQL agent** — queries a local SQLite database of composers, pieces, and listening history 
- **Wikipedia agent** — looks up biographical facts about composers 
- **YouTube agent** — generates a YouTube search link when the user wants to watch or listen to something 

A supervisor routes between agents, and guardrails protect against prompt injection and PII leakage. 

---

## Composers covered 

Ludovico Einaudi, Gibran Alcocer, Yann Tiersen, Virginio Aiello, and Max Richter 

---

## Tech stack

- LangGraph — agent graph and routing 
- LangChain — LLM and embedding integrations 
- Ollama — local LLM server (qwen2.5 + nomic-embed-text)
- Chroma — local vector store 
- SQLite — structured database 
- Docker — containerization
- GitHub Actions - CI/CD pipeline

--- 

## CI/CD Pipeline 

This repo uses GitHub Actions to automatically build and push the Docker image to Docker Hub whenever a version tag is pushed. 

**Trigger:** pushing a tag matching `v*` (e.g. `v1`, `v2`)

**What it does:**
1. Checks out the repository on a GitHub-hosted Ubuntu runner 
2. Logs in to Docker Hub using encrypted GitHub Secrets 
3. Generates image tags and labels from the git tag 
4. Builds the Docker image using the Dockerfile 
5. Pushes the image to Docker Hub as `username/piano-agent:v*` and `username/piano-agent:latest` 

**To trigger the pipeline:** 
```bash
git tag v2 
git push origin v2 
```

**No credentials are stored in the code.** Docker Hub username and access token are stored as encrypted GitHub Secrets and injected at runtime.


---

## Prerequisites 

- Docker Desktop installed and running
- Ollama installed

---

## Running with Docker

### Way 1 — Manual (Dockerfile + image + containers + network)
```bash
#Build the image
docker build -t piano-agent .

#Create the network
docker network create piano-network

#Run the Ollama container
docker run -d --name ollama --network piano-network -p 11434:11434 -v ollama_data:/root/.ollama ollama/ollama:latest

#Pull the needed models
docker exec ollama ollama pull qwen2.5
docker exec ollama ollama pull nomic-embed-text

#Run the app container
docker run -it --name piano-app --network piano-network -e OLLAMA_HOST=http://ollama:11434 -v chroma_data:/app/chroma_db -v piano_db_data:/app/piano_db piano-agent
```

### Way 2 — Docker Compose
```bash
#Start everything
docker compose up -d

#Pull the models (first time only)
docker exec ollama ollama pull qwen2.5
docker exec ollama ollama pull nomic-embed-text

#Attach to the chatbot
docker attach piano-app

#To stop:
docker compose down
```

## Running locally (without Docker)
```bash
#Create and activate virtual environment
python -m venv venv
source venv/bin/activate

#Install dependencies
pip install -r requirements.txt

#Start Ollama in a separate terminal
ollama serve

#Run the app in the activated venv
python piano_multi_agent.py
```

---

## Project structure
```
neoclassical-piano-agent/
├── .github/
│   └── workflows/
│       └── docker-publish.yml  #GitHub Actions CI/CD workflow
├── piano_multi_agent.py        #main application
├── piano_data.txt              #knowledge base for the RAG pipeline
├── requirements.txt            #Python dependencies
├── Dockerfile                  #builds the app image
├── .dockerignore               #excludes unnecessary files from the image
├── docker-compose.yml          #defines services, network, and volumes
├── commands.sh                 #documented terminal commands to run both ways
└── .gitignore
```
