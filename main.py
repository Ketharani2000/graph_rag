import os
import nest_asyncio
import logging

from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from config import config

# Logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
nest_asyncio.apply()

# Setup LLM & Embedding
llm = Ollama(
    model=config.OLLAMA_LLM_MODEL,
    base_url=f"http://{config.OLLAMA_HOST}:{config.OLLAMA_PORT}",
    request_timeout=600.0
)
Settings.llm = llm

embed_model = OllamaEmbedding(
    model_name=config.OLLAMA_EMBED_MODEL,
    base_url=f"http://{config.OLLAMA_HOST}:{config.OLLAMA_PORT}",
    trust_remote_code=True
)
Settings.embed_model = embed_model

# Load data
loader = SimpleDirectoryReader(
    input_dir=config.DOC_DIR,
    required_exts=[".pdf"],
    recursive=True
)
from llama_index.core.schema import Document
import fitz  # PyMuPDF

docs = []
for filename in os.listdir(config.DOC_DIR):
    if filename.endswith(".pdf"):
        path = os.path.join(config.DOC_DIR, filename)
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc[5:30])  
        docs.append(Document(text=text, metadata={"source": filename}))


# Neo4j Graph Store setup
graph_store = Neo4jGraphStore(
    username=config.NEO4J_USERNAME,
    password=config.NEO4J_PASSWORD,
    url=config.NEO4J_URI,
    database="neo4j",
    timeout=600.0
)

# Path to persist index
PERSIST_DIR = "./graph_index_storage"

# Check if index exists, reuse or build
if os.path.exists(PERSIST_DIR):
    # Load from disk
    storage_context = StorageContext.from_defaults(
        graph_store=graph_store,
        persist_dir=PERSIST_DIR
    )
    kg_index = KnowledgeGraphIndex(storage_context=storage_context, embed_model=embed_model)
    print("✅ Reused persisted knowledge graph index.")
else:
    # Create new and persist
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    kg_index = KnowledgeGraphIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embed_model,
        max_triplets_per_chunk=8,
        show_progress=True
    )
    storage_context.persist(persist_dir=PERSIST_DIR)
    print("📦 Built and persisted new knowledge graph index.")

# Setup retriever and query engine
graph_rag_retriever = KnowledgeGraphRAGRetriever(
    storage_context=storage_context,
    verbose=True,
)

query_engine = RetrieverQueryEngine.from_args(
    graph_rag_retriever,
    embed_model=embed_model,
)

# Query loop
while (user_query := input("\n\nWhat do you want to know about these files?\n")):
    response = query_engine.query(user_query)
    print(str(response))
