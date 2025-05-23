# Barebone Graph RAG with llama-index and local Ollama
This is the barebone Graph RAG using Ollama + llama-index 

## neo4j
This template uses neo4j as Graph Store, embedding and properties are created by KnowledgeGraphIndex.

A free neo4j instance is used, please note a fair use manner is expected, and the instance
can be taken offline if abused. 

RetrieverQueryEngine is used to retrieve graph based knowledge from the graph db.


## Vector Store (used for vector index, without Graph index)
Either Qdrant or Redis can be used. 

Additionally we can also just use local index

vector index allow embeding search, and answer questions that are "sounds relevant". 
