import os
import nest_asyncio
import logging
import fitz

from llama_index.core import Settings, Document
from llama_index.core import StorageContext
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from config import config

# === Setup Logging ===
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
nest_asyncio.apply()

# === Setup LLM & Embedding ===
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

# === Load PDF ===
specific_filename = "/workspace/OllamaGraphRAGPoC/input_dir_2/sap-hana-on-vmware-vsphere-bp_0.pdf"
file_path = os.path.join(config.DOC_DIR, specific_filename)

docs = []
if os.path.exists(file_path):
    doc = fitz.open(file_path)
    text = "\n".join(page.get_text() for page in doc)
    docs.append(Document(text=text, metadata={"source": specific_filename}))
else:
    raise FileNotFoundError(f"File not found: {file_path}")

# === Neo4j Knowledge Graph Store ===
graph_store = Neo4jGraphStore(
    username=config.NEO4J_USERNAME,
    password=config.NEO4J_PASSWORD,
    url=config.NEO4J_URI,
    database="neo4j",
    timeout=600.0
)

# === Property Graph Index Setup ===
PERSIST_DIR = "./graph_index_storage"

# NOTE: PropertyGraphIndex does NOT support from_storage yet.
# So we always rebuild it for now.
storage_context = StorageContext.from_defaults(graph_store=graph_store)

kg_index = PropertyGraphIndex.from_documents(
    docs,
    storage_context=storage_context,
    embed_model=embed_model,
    max_triplets_per_chunk=8,
    show_progress=True
)

print("Built fresh property graph index.")

# === Vector Index for Text Embeddings ===
vector_index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)

# === Create Individual Retrievers ===
kg_retriever = KnowledgeGraphRAGRetriever(
    index=kg_index,
    storage_context=storage_context,
    verbose=True,
)

vector_retriever = vector_index.as_retriever(similarity_top_k=5)

# === Combine Retrievers Manually ===
class ManualHybridRetriever:
    def __init__(self, retrievers, weights):
        assert len(retrievers) == len(weights), "Mismatch in retrievers and weights"
        self.retrievers = retrievers
        self.weights = weights

    def retrieve(self, query):
        results = []
        for retriever, weight in zip(self.retrievers, self.weights):
            retrieved = retriever.retrieve(query)
            for node in retrieved:
                node.score = (node.score or 1.0) * weight
                results.append(node)
        results.sort(key=lambda x: x.score, reverse=True)
        return results

hybrid_retriever = ManualHybridRetriever(
    retrievers=[kg_retriever, vector_retriever],
    weights=[0.6, 0.4]
)

# === Wrap hybrid retriever in query engine ===
class ManualHybridRetrieverWrapper:
    def __init__(self, retriever):
        self.retriever = retriever

    def retrieve(self, query):
        return self.retriever.retrieve(query)

query_engine = RetrieverQueryEngine.from_args(
    retriever=ManualHybridRetrieverWrapper(hybrid_retriever),
    embed_model=embed_model,
)

# === Chain-of-Thought Prompting ===
def apply_cot_prompting(query: str) -> str:
    cot_instruction = (
        "Please think step-by-step and explain your reasoning clearly. "
        "Break the explanation into logical parts before giving your final answer.\n\n"
    )
    return cot_instruction + query

# === Query Loop ===
while (user_query := input("\n\nWhat do you want to know about these files?\n")):
    cot_query = apply_cot_prompting(user_query)
    response = query_engine.query(cot_query)
    print(str(response))
