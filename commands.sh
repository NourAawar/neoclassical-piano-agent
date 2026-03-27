#This file contains all the terminal commands needed to run the piano_multi_agent system using two ways:
#   1) Dockerfile, docker image, docker containers, and docker networks (manual)
#   2) Docker Compose 

#Note: This is written as a shell script but it is meant to be read and run command by command (not executed all at once). 


#Some prerequisites: 

#Verify Docker is installed and running: 
docker --version 

#Verify that Ollama is installed: 
ollama --version 


#Way 1: Dockerfile, docker image, docker containers, and docker networks (manual)

#Step 1: We build the Docker image.
docker build -t piano-agent . 

#Verify that the image was created: 
docker image ls 

#Step 2: We create the network.  
docker network create piano-network 

#Step 3: Run the Ollama container. 
docker run -d --name ollama --network piano-network -p 11434:11434 -v ollama_data:/root/.ollama ollama/ollama:latest 

#Step 4: We pull the models into the Ollama container. 
docker exec ollama ollama pull qwen2.5
docker exec ollama ollama pull nomic-embed-text

#Verify that both models are downloaded:
docker exec ollama ollama list

#Step 5: We run the piano app container. 
docker run -it --name piano-app --network piano-network -e OLLAMA_HOST=http://ollama:11434 -v chroma_data:/app/chroma_db -v piano_db_data:/app/piano_db piano-agent

#Some debugging commands: 

#We check all running containers: 
docker container ps 

#We can check the logs of the app container: 
docker logs piano-app 

#To open a shell inside the running app container: 
docker exec -it piano-app bash 

#Cleanup: 

#Stop both containers: 
docker stop piano-app ollama 

#Remove both containers: 
docker rm piano-app ollama 

#Remove the network: 
docker network rm piano-network 


#Way 2: Docker Compose 

#We start everything in the background (that way, the terminal stays free): 
docker compose up -d

#Then, we pull the models into the running ollama container
docker exec ollama ollama pull qwen2.5
docker exec ollama ollama pull nomic-embed-text

#Next, we attach the piano-app container to use the chatbot (since everything was started in the background): 
docker attach piano-app

#We force rebuild the image (if piano_multi_agent.py changed): 
docker compose up --build

#Then, check the status:
docker compose ps 

#We can check the logs: 
docker compose logs 

#We can also check the logs of a specific service:
docker compose logs piano-app

#Stop everything: 
docker compose down 

#To stop everything and delete all volumes, use this instead: 
docker compose down -v 

