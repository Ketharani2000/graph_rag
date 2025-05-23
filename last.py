import nest_asyncio
import os
import re
from pathlib import Path
from typing import List, Dict
from llama_index.core import Settings, SimpleDirectoryReader, PromptTemplate
from llama_index.core import StorageContext, ServiceContext
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import ImageNode, TextNode
from transformers import pipeline
from config import config
import logging
import fitz  # PyMuPDF
import requests
from sentence_transformers import SentenceTransformer
import hdbscan
import numpy as np
from requests.adapters import HTTPAdapter  # Missing import
from urllib.parse import urlparse  # Add if not present



# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nest_asyncio.apply()

# --- Constants ---
STORAGE_DIR = "./storage"
Path(STORAGE_DIR).mkdir(exist_ok=True)

# --- Document Expansion ---
class SAPDocumentExpander:
    def __init__(self):
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=3))
        self.session.mount("http://", HTTPAdapter(max_retries=3)) 
        
    def _has_credentials(self) -> bool:
        return os.path.exists(config.SAP_CREDENTIALS_FILE)
        
    def expand(self, text: str) -> List[Dict]:
        urls = re.findall(r'https?://\S+', text)
        expanded_docs = []
        
        for url in urls:
            try:
                if "sap.com/note" in url and not self._has_credentials():
                    raise PermissionError("SAP Note requires authentication")
                    
                response = self.session.get(url, timeout=10)
                expanded_docs.append({
                    "content": response.text,
                    "metadata": {"source": url}
                })
            except Exception as e:
                logger.warning(f"Failed to expand {url}: {str(e)}")
                expanded_docs.append({
                    "content": f"External document unavailable: {str(e)}",
                    "metadata": {"source": url, "error": True}
                })
        return expanded_docs

# --- Image Processing ---
class TechnicalDiagramAnalyzer:
    def __init__(self):
        self.vqa = pipeline("visual-question-answering", 
                          model="Salesforce/blip2-opt-2.7b")
        
    def describe(self, image) -> str:
        return self.vqa(image, "Describe this technical diagram in detail, focusing on components and relationships.")

# --- Concept Mapping ---
class SAPConceptMapper:
    def __init__(self):
        self.encoder = SentenceTransformer('all-mpnet-base-v2')
        
    def cluster(self, texts: List[str]) -> Dict:
        embeddings = self.encoder.encode(texts)
        clusters = hdbscan.HDBSCAN(min_cluster_size=3).fit(embeddings)
        return self._create_concept_nodes(clusters, texts, embeddings)

    def _create_concept_nodes(self, clusters, texts, embeddings):
        concepts = {}
        for idx, (text, cluster_id) in enumerate(zip(texts, clusters.labels_)):
            if cluster_id == -1:
                continue  # Noise
            if cluster_id not in concepts:
                concepts[cluster_id] = {
                    "label": f"Concept_{cluster_id}",
                    "examples": [],
                    "embedding": embeddings[idx]
                }
            concepts[cluster_id]["examples"].append(text)
        return concepts

# --- Core Pipeline ---
def process_pdfs() -> tuple:
    # 1. Load and expand documents
    loader = SimpleDirectoryReader(input_dir=config.DOC_DIR, 
                                  required_exts=[".pdf"], 
                                  recursive=True)
    docs = loader.load_data()
    
    # Expand linked content
    expander = SAPDocumentExpander()
    for doc in docs[:]:  # Iterate copy to allow modification
        expanded = expander.expand(doc.text)
        docs.extend([TextNode(text=e["content"], metadata=e["metadata"]) 
                    for e in expanded])

    # 2. Process images
    diagram_analyzer = TechnicalDiagramAnalyzer()
    image_nodes = []
    for doc_path in [os.path.join(config.DOC_DIR, f) for f in os.listdir(config.DOC_DIR)]:
        for img in extract_images(doc_path):
            description = diagram_analyzer.describe(img)
            image_nodes.append(ImageNode(
                text=description,
                image=img.tobytes(),
                metadata={"source": doc_path}
            ))
    
    # 3. Concept Mapping
    concept_mapper = SAPConceptMapper()
    text_chunks = [doc.text for doc in docs if isinstance(doc, TextNode)]
    concepts = concept_mapper.cluster(text_chunks)
    
    # Create concept nodes
    concept_nodes = [
        TextNode(
            text=f"Concept: {c['label']}\nExamples: {'; '.join(c['examples'][:3])}",
            embedding=c["embedding"],
            metadata={"type": "concept"}
        ) for c in concepts.values()
    ]
    
    return docs + image_nodes + concept_nodes

# --- Knowledge Graph Setup ---
def get_kg_index(nodes: list) -> PropertyGraphIndex:
    # Configure Neo4j
    graph_store = Neo4jGraphStore(
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        url=config.NEO4J_URI,
        database="neo4j",
        timeout=600.0
    )
    
    # Add performance optimization rules
    graph_store.query("""
    CREATE CONSTRAINT IF NOT EXISTS FOR (p:Parameter) REQUIRE p.name IS UNIQUE
    """)
    
    # Create/load index
    if os.path.exists(f"{STORAGE_DIR}/graph_index"):
        return PropertyGraphIndex.load_from_disk(f"{STORAGE_DIR}/graph_index")
    
    # Build new index
    Settings.chunk_size = 1024  # Larger chunks for technical docs
    Settings.chunk_overlap = 100
    Settings.num_workers = 4
    
    index = PropertyGraphIndex(
        nodes=nodes,
        storage_context=StorageContext.from_defaults(graph_store=graph_store),
        embed_model=OllamaEmbedding(model_name="nomic-embed-text"),
        include_images=True
    )
    
    # Add parameter relationships
    conflict_rules = [
        ("sched.vcpuXx.affinity", "numa.nodeAffinity", "CONFLICTS_WITH"),
        ("sched.nodeX.affinity", "numa.nodeAffinity", "REQUIRES_VALIDATION")
    ]
    for param1, param2, rel_type in conflict_rules:
        graph_store.upsert_triplet(param1, rel_type, param2)
    
    index.persist(f"{STORAGE_DIR}/graph_index")
    return index

# --- Query Engine ---
def create_query_engine(index: PropertyGraphIndex) -> RetrieverQueryEngine:
    # Custom prompt template
    qa_prompt = PromptTemplate("""Context:
    {context_str}

    Analysis Steps:
    1. Identify key components from [VMware, SAP HANA, vSphere, Guest OS]
    2. Check parameter conflicts in knowledge graph
    3. Review performance optimization patterns
    4. Synthesize recommendations

    Question: {query_str}
    Answer:""")
    
    # Configure retrievers
    retriever = index.as_retriever(
        similarity_top_k=3,
        include_text=True,
        image_similarity_top_k=1
    )
    
    return index.as_query_engine(
        retriever=retriever,
        response_mode="tree_summarize",
        text_qa_template=qa_prompt,
        verbose=True
    )

# --- Main Execution ---
if __name__ == "__main__":
    # Process documents
    nodes = process_pdfs()
    
    # Build/load knowledge graph
    kg_index = get_kg_index(nodes)
    
    # Initialize query engine
    query_engine = create_query_engine(kg_index)
    
    # Interactive session
    while (query := input("\nAsk about SAP HANA on VMware (type 'exit' to quit): ")):
        if query.lower() == 'exit':
            break
            
        try:
            response = query_engine.query(query)
            print(f"\nResponse:\n{response}")
            
            # Display conflicts
            if "parameter" in query.lower():
                conflicts = kg_index.graph_store.query(
                    "MATCH (p1)-[:CONFLICTS_WITH]->(p2) RETURN p1.name, p2.name"
                )
                if conflicts:
                    print("\n Parameter Conflicts:")
                    for row in conflicts:
                        print(f"- {row['p1.name']} vs {row['p2.name']}")
                        
        except Exception as e:
            print(f"Error: {str(e)}")
            logger.error(f"Query failed: {query}", exc_info=True)
