#We will use the official Python 3.11 slim image as the base. 
#"Slim" means it contains only the minimum packages needed to run Python with no build tools, no docs, and no extras. 
#This keeps the final image smaller. 
FROM python:3.11-slim

#All the subsequent commands (COPY, RUN, and CMD) will be relative to /app inside the container. 
#If /app dpesn't exist, Docker creates it automatically. 
WORKDIR /app

#ChromaDB (the vector store) requires some C build tools to compile its native extensions. 
#We will install them here before installing Python packages. 
#--no-install-recommends keeps the layer small by skipping optional packages. 
#We clean up the apt cache in the same RUN command to avoid creating a separate layer that would inflate the image size. 
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

#Then, we copy files from our local machine into the container image. 
#We will copy requirements.txt before the rest of the code which is practical for caching since if it hasn't changed, 
#Docker reuses the cached pip install layer on the next build and thus saves time.
COPY requirements.txt .

#After that, we install all the Python packages listed in requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

#We copy our main Python application file into /app inside the image. 
COPY piano_multi_agent.py .

#We copy the knowledge base text file that the app uses to build the Chroma vector store. 
#Without this step, init_vector_store() would fail because it tries to load "piano_data.txt" with TextLoader. 
COPY piano_data.txt . 

#Now we set the environment variable that will be available inside the container at runtime. 
ENV OLLAMA_HOST=http://ollama:11434

#Then, we run our application file.
CMD ["python", "-u", "piano_multi_agent.py"]
