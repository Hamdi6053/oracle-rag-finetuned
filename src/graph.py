# graph.py - LangGraph Compatible RAG System (Fixed)

import os
import re
import requests
import time
import json
import webbrowser
from typing import List, Dict, Any
from dotenv import load_dotenv
from typing_extensions import NotRequired

# LangChain and LangGraph libraries
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.embeddings import OllamaEmbeddings
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_chroma import Chroma
import pickle
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnableLambda
from langchain_ollama import OllamaLLM

# Basit web search - DeepAgents kaldırıldı

# Advanced Retrieval Technologies (from 2_query.py)
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnableLambda
from langchain_ollama import OllamaLLM
# Your custom service - Fixed import


# =============================================================================
# BASIC SETUP AND CHECKS
# =============================================================================

OLLAMA_API_URL = "http://localhost:11434"

def is_ollama_running():
    try:
        response = requests.get(OLLAMA_API_URL)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

if not is_ollama_running():
    print("❌ ERROR: Ollama service is not running. Please start Ollama.")
    exit()
else:
    print("✅ Ollama service is active.")

# Model settings
LOCAL_LLM = "Ahmet_Hamdi/oracle_final_deepseek:latest"
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en", 
                                       model_kwargs={'device': 'cpu'},
                                       encode_kwargs={'normalize_embeddings': True})
ORACLE_COLLECTION_NAME = "oracle_md_enhanced_v2"  # 1_ingestion_cleaned.py ile aynı isim

# Advanced Retrieval Configuration (from 2_query.py)
ENABLE_MULTIQUERY = True          # MultiQuery expansion aktif
ENABLE_CROSS_ENCODER = True       # Cross-encoder reranking aktif
RERANK_TOP_N = 8                  # Cross-encoder'ın seçeceği chunk sayısı
DIVERSITY_FACTOR = True           # Parent çeşitliliğini artırma
ENABLE_DEBUG_INFO = True          # Debug açık - sorun tespiti için

print(f"Using LLM: {LOCAL_LLM}")
print(f"Using Embedding: BAAI/bge-base-en")
print(f"Database Collection: {ORACLE_COLLECTION_NAME}")

# Advanced Retrieval Configuration Display
print(f"🔧 Advanced Pipeline: MultiQuery+CrossEncoder+ParentMapping {'✅' if ENABLE_MULTIQUERY and ENABLE_CROSS_ENCODER else '⚠️'}")

# =============================================================================
# CHROMA VECTORDB CONNECTION AND RETRIEVER SETUP
# =============================================================================

print("🔧 Setting up embedding model and Chroma VectorDB connection...")

try:
    # Embedding modeli test et
    embedding_model = EMBEDDING_MODEL
    embedding_model.embed_query("test")  # Quick test
    print("✅ Embedding function loaded successfully.")
except Exception as e:
    print(f"❌ ERROR: Could not load embedding function: {e}")
    exit()

try:
    # Chroma VectorDB bağlantısı (PgvectorService yerine)
    vectorstore = Chroma(
        collection_name=ORACLE_COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory="./oracle_vector_db_md_enhanced"  # Yeni Chroma dizini
    )
    print("✅ Chroma VectorStore başarıyla bağlandı.")
except Exception as e:
    print(f"❌ ERROR: Could not connect to Chroma VectorStore: {e}")
    exit()

def setup_advanced_retriever():
    """2_query.py'deki gelişmiş retriever pipeline'ını kurar"""
    try:
        # Base retriever (2_query.py'deki gibi)
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
        print("📋 Base retriever kuruldu (k=20)")
        
        current_retriever = base_retriever
        
        if ENABLE_MULTIQUERY:
            try:
                # MultiQuery expansion için LLM (2_query.py'deki ayarlar)
                expansion_llm = OllamaLLM(model="gemma3:4b", temperature=0.1)
                multi_query_retriever = MultiQueryRetriever.from_llm(
                    retriever=base_retriever, 
                    llm=expansion_llm
                )
                current_retriever = multi_query_retriever
                print("✅ MultiQuery retriever aktif (5 alternatif sorgu)")
            except Exception as e:
                print(f"⚠️ MultiQuery kurulamadı: {e}, base retriever kullanılacak")
        
        if ENABLE_CROSS_ENCODER:
            try:
                # Cross-encoder reranking (2_query.py'deki model)
                cross_encoder_model = HuggingFaceCrossEncoder(
                    model_name="BAAI/bge-reranker-base"
                )
                compressor = CrossEncoderReranker(
                    model=cross_encoder_model, 
                    top_n=RERANK_TOP_N
                )
                enhanced_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor, 
                    base_retriever=current_retriever
                )
                print(f"✅ Cross-encoder reranking aktif (top-{RERANK_TOP_N})")
                return enhanced_retriever
            except Exception as e:
                print(f"⚠️ Cross-encoder kurulamadı: {e}, MultiQuery kullanılacak")
                return current_retriever
        
        return current_retriever
        
    except Exception as e:
        print(f"❌ Gelişmiş retriever kurulamadı: {e}")
        # Fallback to simple MMR retriever
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 7, 'fetch_k': 25, 'lambda_mult': 0.6}
        )

try:
    retriever = setup_advanced_retriever()
    print(f"✅ '{ORACLE_COLLECTION_NAME}' koleksiyonu gelişmiş retriever ile hazır.")
except Exception as e:
    print(f"❌ ERROR: Could not create retriever: {e}")
    exit()

# =============================================================================
class SimplePickleStore:
    """Basit pickle tabanlı kalıcı depolama - graph.py için"""
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.store_file = os.path.join(storage_path, "documents.pkl")
        self._store = self._load_store()
    
    def _load_store(self) -> Dict:
        if os.path.exists(self.store_file):
            try:
                with open(self.store_file, 'rb') as f:
                    return pickle.load(f)
            except:
                print("⚠️ Store dosyası okunamadı, yeni store oluşturuluyor.")
        return {}
    
    def _save_store(self):
        with open(self.store_file, 'wb') as f:
            pickle.dump(self._store, f)
    
    def mget(self, keys: List[str]) -> List[Document]:
        return [self._store.get(key) for key in keys if key in self._store]
    
    def get_parent_document(self, parent_id: str) -> Document:
        return self._store.get(parent_id)

# Parent document store setup
try:
    parent_store = SimplePickleStore("./parent_document_store_md_enhanced")
    print("✅ Parent document store connected successfully.")
except Exception as e:
    print(f"⚠️ Parent store error: {e}")
    parent_store = None

def get_parent_context(child_docs: List[Document]) -> str:
    """Child dokümanlarından parent context'i getir"""
    if not parent_store or not child_docs:
        return ""
    
    parent_contexts = []
    seen_parents = set()
    
    for doc in child_docs:
        parent_id = doc.metadata.get('parent_id')
        if parent_id and parent_id not in seen_parents:
            parent_doc = parent_store.get_parent_document(parent_id)
            if parent_doc:
                parent_contexts.append(f"[PARENT CONTEXT]\n{parent_doc.page_content[:500]}...\n")
                seen_parents.add(parent_id)
    
    return "\n".join(parent_contexts)

def get_final_parent_documents(child_docs: List[Document]) -> Dict:
    """
    2_query.py'deki EXACT parent document mapping mantığı:
    Gelişmiş parent mapping: Çeşitlilik sağlar ve aynı parent'tan gelen
    farklı chunk'ları birleştirir
    """
    if not parent_store or not child_docs:
        return {'final_docs': child_docs, 'details': []}

    parent_docs_map = {}
    parent_chunk_counts = {}  # Her parent'tan kaç chunk geldiğini takip et
    
    if ENABLE_DEBUG_INFO:
        print(f"\n🏗️ Parent Document Mapping Detayları:")
        print(f"📊 Toplam {len(child_docs)} child chunk işleniyor...")
    
    # İlk geçiş: Her parent için en iyi chunk'ı seç (2_query.py mantığı)
    for child_doc in child_docs:
        parent_id = child_doc.metadata.get('parent_id')
        if parent_id and parent_id in parent_store._store:
            relevance_score = child_doc.metadata.get('relevance_score', 0.0)
            
            if parent_id not in parent_docs_map or relevance_score > parent_docs_map[parent_id].metadata.get('relevance_score', 0.0):
                parent_doc = parent_store.get_parent_document(parent_id)
                if parent_doc:
                    new_metadata = parent_doc.metadata.copy()
                    new_metadata['relevance_score'] = relevance_score
                    new_metadata['best_chunk_score'] = relevance_score
                    parent_docs_map[parent_id] = Document(page_content=parent_doc.page_content, metadata=new_metadata)
                    
                    # Parent chunk sayısını takip et
                    if parent_id not in parent_chunk_counts:
                        parent_chunk_counts[parent_id] = 0
                    parent_chunk_counts[parent_id] += 1
        else:
            # Parent ID yoksa child'ı direkt kullan
            no_parent_id = f"no_parent_{len(parent_docs_map)}"
            child_doc.metadata['relevance_score'] = child_doc.metadata.get('relevance_score', 0.0)
            parent_docs_map[no_parent_id] = child_doc
    
    # Parent çeşitliliğini artır: Aynı parent'tan gelen chunk'ları birleştir (2_query.py)
    if ENABLE_DEBUG_INFO:
        print(f"\n🔗 Parent Chunk Birleştirme:")
    for parent_id, chunk_count in parent_chunk_counts.items():
        if chunk_count > 1:
            if ENABLE_DEBUG_INFO:
                print(f"   📚 Parent {parent_id[:8]}...: {chunk_count} chunk bulundu")
            
            # Bu parent'tan gelen tüm chunk'ları bul
            parent_chunks = [doc for doc in child_docs if doc.metadata.get('parent_id') == parent_id]
            
            # En yüksek skorlu chunk'ı seç ve diğerlerinden ek bilgi ekle
            best_chunk = max(parent_chunks, key=lambda x: x.metadata.get('relevance_score', 0.0))
            parent_doc = parent_docs_map[parent_id]
            
            # Ek bilgileri metadata'ya ekle
            parent_doc.metadata['total_chunks_from_parent'] = chunk_count
            parent_doc.metadata['chunk_diversity'] = len(set(doc.metadata.get('header_hierarchy', '') for doc in parent_chunks))
    
    # Sonuçları skor ve çeşitliliğe göre sırala (2_query.py mantığı)
    final_docs = sorted(
        parent_docs_map.values(), 
        key=lambda doc: (
            doc.metadata.get('relevance_score', 0.0),  # Önce skor
            doc.metadata.get('chunk_diversity', 1),    # Sonra çeşitlilik
            doc.metadata.get('total_chunks_from_parent', 1)  # Son olarak chunk sayısı
        ), 
        reverse=True
    )
    
    if ENABLE_DEBUG_INFO:
        print(f"\n📊 Parent Mapping Sonuçları:")
        print(f"   🔍 Toplam unique parent: {len(final_docs)}")
        for i, doc in enumerate(final_docs, 1):
            parent_id = doc.metadata.get('parent_id', 'unknown')[:8]
            score = doc.metadata.get('relevance_score', 0.0)
            diversity = doc.metadata.get('chunk_diversity', 1)
            total_chunks = doc.metadata.get('total_chunks_from_parent', 1)
            print(f"   {i}. Parent {parent_id}... | Skor: {score:.4f} | Çeşitlilik: {diversity} | Chunk: {total_chunks}")
    
    details = [{'chunk_index': i + 1, **doc.metadata, 'full_content': doc.page_content, 'content_preview': doc.page_content[:200] + "..."} for i, doc in enumerate(final_docs)]
    return {'final_docs': final_docs, 'details': details}

# =============================================================================
# LLM AND PROMPT SETUP
# =============================================================================

try:
    # Base deterministic model
    llm_json = ChatOllama(model=LOCAL_LLM, format="json", temperature=0)
    llm_text = ChatOllama(model=LOCAL_LLM, temperature=0)

    # Additional models for hallucination voting (can reuse same model with different seeds/temps or different local models)
    HALLUCINATION_MODELS = [
        ChatOllama(model=LOCAL_LLM, format="json", temperature=0),
        ChatOllama(model=LOCAL_LLM, format="json", temperature=0.1),
        ChatOllama(model=LOCAL_LLM, format="json", temperature=0.2),
    ]
    print(f"✅ {LOCAL_LLM} models loaded for generation and hallucination voting.")
except Exception as e:
    print(f"❌ ERROR: Could not load LLM: {e}")
    exit()

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    
    datasource: str = Field(
        description="The datasource to use, either 'vectorstore' or 'web_search'.",
        enum=["vectorstore", "web_search"]
    )
    reasoning_category: str = Field(
        description="The category of reasoning for the routing decision.",
        enum=[
            "General SCM Question",
            "Specific Technical Question",
            "Latest Information Request",
            "General Knowledge Query"
        ]
    )
    reasoning: str = Field(
        description="A brief explanation of why the query was routed to this datasource."
    )

# 1. QUERY ROUTER - Oracle Manufacturing-Focused with Technical Detection
router_prompt = PromptTemplate(
    template="""<purpose>
You are an expert at routing a user question to a vectorstore or web search. The vectorstore contains documentation about Oracle SCM, focusing on general information for consultants, including UI, module foundations, and general SCM processes. It does not contain specific error codes, detailed technical implementation steps, or the latest product updates. Use the vectorstore for questions that can be answered with this general documentation. Otherwise, use web search.
</purpose>

<routing_intelligence>
You must make a decision based on the following criteria:

1.  **General SCM Questions (vectorstore):** If the question is about a general Oracle SCM topic, process, or functionality that would be covered in standard documentation, route to the vectorstore.
    -   *Examples: "What is Oracle Manufacturing?", "How are production orders created in Oracle Manufacturing?", "Explain the procure-to-pay process in Oracle SCM."*

2.  **Specific Technical Questions (web_search):** If the question is highly technical, asks for specific implementation steps, error code explanations, or details not found in general documentation, route to web search.
    -   *Examples: "How do I fix ORA-00942 in my custom Oracle report?", "What are the exact API parameters for creating a purchase order in Oracle Fusion SCM?", "Show me the SQL query to find all open work orders."*

3.  **Latest Information Requests (web_search):** If the question asks for the latest updates, recent news, or information about current events, route to web search.
    -   *Examples: "What are the new features in the latest Oracle SCM Cloud update?", "Is there a known issue with Oracle SCM in Q2 2024?"*

4.  **General Knowledge Queries (web_search):** If the question is not related to Oracle SCM or is a general knowledge question, route to web search.
    -   *Examples: "What is the capital of France?", "Who won the last World Cup?"*
</routing_intelligence>

<output_format>
{format_instructions}
</output_format>

Question: {question}""",
    input_variables=["question"],
    partial_variables={"format_instructions": PydanticOutputParser(pydantic_object=RouteQuery).get_format_instructions()},
)

question_router = router_prompt | llm_json | PydanticOutputParser(pydantic_object=RouteQuery)
# 2. RETRIEVAL GRADER - Oracle SCM Relevance Scoring System
retrieval_grader_prompt = PromptTemplate(
    template="""You are an Oracle SCM expert acting as a second-level evaluator to decide if a retrieved document should be used to answer a query. You must be strict and catch vector similarity errors.

<known_vector_issues>
1) Context sensitivity: Documents can be semantically similar but not about the same Oracle topic. Reject if the key entities/modules in the question are missing from the document.
2) Unrelated noise: Retrieved text may contain general ERP content without Oracle relevance. Reject if Oracle context is weak or absent.
3) Temporal/vendor mismatch: If the question implies Oracle Fusion/Cloud but document is only about legacy EBS (11i/R12) or non-Oracle products (SAP/Dynamics), treat as not relevant unless the question explicitly asks about that legacy/vendor.
</known_vector_issues>

<accept_criteria>
Return {{"score": "yes"}} only if ALL are true:
- Strong Oracle context: Oracle terms or modules relevant to the question are explicitly present (e.g., INV, BOM, WIP, MRP, Procurement, Fusion, Cloud, API, concurrent program).
- Topic alignment: The document addresses the same subject the user asked about (procedures, definitions, configurations, troubleshooting) rather than adjacent but different topics.
- Answerability signal: The excerpt contains information that could help answer the question (definitions, steps, parameters, tables, forms, reports, APIs, configurations).

Otherwise return {{"score": "no"}}.

<reject_signals>
- Mentions of non-Oracle ERP (SAP, Microsoft Dynamics, NetSuite) unless the question requests comparisons.
- Only generic IT/ERP content without Oracle context.
- Legacy-only (EBS 11i/R12) content when question implies Fusion/Cloud and does not ask for EBS.
- Mismatch of key entities/modules between question and document.
</reject_signals>

Document excerpt:
{document}

User question:
{question}

Respond only in JSON: {{"score": "yes"}} or {{"score": "no"}}""",
    input_variables=["question", "document"],
)

retrieval_grader = retrieval_grader_prompt | llm_json | JsonOutputParser()

# Enhanced fallback grader with Oracle SCM focus
def oracle_grader_fallback(question, document_content):
    """Enhanced fallback grader for Oracle SCM relevance"""
    doc_lower = document_content.lower()
    question_lower = question.lower()
    
    # High-priority Oracle SCM terms (higher weight)
    oracle_priority_terms = [
        "oracle", "manufacturing", "inventory", "production order", "work order", 
        "bom", "routing", "wip", "mrp", "mps", "procurement", "purchasing",
        "supplier", "quality", "costing", "wms", "warehouse", "receiving",
        "shipping", "planning", "forecast", "concurrent program", "flexfield"
    ]
    
    # Technical Oracle terms (medium weight)  
    oracle_technical_terms = [
        "e-business suite", "ebs", "r12", "11i", "fusion", "cloud",
        "api", "interface", "workflow", "approval", "profile option",
        "responsibility", "concurrent", "request", "form", "report"
    ]
    
    # General business terms (low weight)
    business_terms = [
        "erp", "supply chain", "logistics", "distribution", "operations",
        "finance", "accounting", "process", "system", "application"
    ]
    
    priority_score = sum(1 for term in oracle_priority_terms if term in doc_lower or term in question_lower)
    technical_score = sum(0.7 for term in oracle_technical_terms if term in doc_lower or term in question_lower)  
    business_score = sum(0.3 for term in business_terms if term in doc_lower or term in question_lower)
    
    total_score = priority_score + technical_score + business_score
    
    # Oracle SCM context - be more inclusive for potential Oracle content
    if total_score >= 1.0:  # Lower threshold for Oracle context
        return {"score": "yes"}
    elif any(term in doc_lower for term in ["oracle", "manufacturing", "erp"]):
        return {"score": "yes"}  # Keep potential Oracle content
    else:
        return {"score": "yes"}  # Default inclusive for Oracle environment

# ----------------------------
# Utility Functions from 2_query.py
# ----------------------------

def _dedupe_docs(docs: List[Document]) -> List[Document]:
    """
    Gelişmiş deduplication: İçerik + parent_id + header_hierarchy kombinasyonuna göre
    Farklı parent'lardan gelen benzer içerikli chunk'ları korur
    """
    seen_combinations = set()
    unique_docs = []
    
    for doc in docs:
        # Daha detaylı deduplication için parent_id ve header_hierarchy'yi de dahil et
        parent_id = doc.metadata.get('parent_id', 'unknown')
        header_hierarchy = doc.metadata.get('header_hierarchy', 'unknown')
        content_preview = doc.page_content[:100]  # İlk 100 karakter
        
        # Benzersiz kombinasyon oluştur
        combination = f"{parent_id}|{header_hierarchy}|{content_preview}"
        
        if combination not in seen_combinations:
            seen_combinations.add(combination)
            unique_docs.append(doc)
            if ENABLE_DEBUG_INFO:
                print(f"🔍 Yeni benzersiz chunk: Parent={parent_id[:8]}... | Header={header_hierarchy[:30]}...")
        else:
            if ENABLE_DEBUG_INFO:
                print(f"🔄 Yinelenen chunk kaldırıldı: Parent={parent_id[:8]}... | Header={header_hierarchy[:30]}...")
    
    if ENABLE_DEBUG_INFO:
        print(f"🧹 Deduplication sonrası: {len(docs)} -> {len(unique_docs)} chunk")
    return unique_docs

# ----------------------------
# Advanced Heuristic Evaluator (from 2_query.py)
# ----------------------------

_STOPWORDS = {
    "the","is","are","a","an","of","and","or","to","in","for","on","by","with","at",
    "from","as","that","this","these","those","it","its","be","can","how","what","why",
    "when","where","which","who","whom","about","into","over","under","between","within",
}

_NEGATIVE_ERP_TERMS = {"sap", "microsoft dynamics", "d365", "dynamic 365", "netsuite"}
_LEGACY_TERMS = {"11i", "r12", "e-business suite", "ebs"}
_FUSION_TERMS = {"fusion", "cloud"}
_ORACLE_MODULE_TERMS = {
    "oracle","manufacturing","inventory","inv","bom","routing","wip","mrp","mps","procurement",
    "purchasing","po","supplier","quality","costing","wms","warehouse","receiving","shipping",
    "planning","forecast","concurrent program","flexfield","api","interface","table","report",
}
_ANSWERABILITY_CUES = {"step","steps","click","navigate","setup","configure","parameter","field",
                       "table","view","form","report","run","submit","api","endpoint","payload",
                       "request","response","example","code","sql","procedure","how to"}

def _tokenize_for_overlap(text: str) -> List[str]:
    lowered = text.lower()
    tokens = re.findall(r"[a-zA-Z0-9_\-]+", lowered)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]

def _contains_any(haystack_lower: str, needles: set) -> bool:
    return any(term in haystack_lower for term in needles)

def heuristic_relevance_signals(question: str, document_text: str) -> Dict[str, Any]:
    """2_query.py'deki gelişmiş heuristic relevance scoring"""
    ql = question.lower()
    dl = document_text.lower()

    q_tokens = set(_tokenize_for_overlap(ql))
    d_tokens = set(_tokenize_for_overlap(dl))
    overlap = q_tokens.intersection(d_tokens)
    overlap_ratio = (len(overlap) / max(1, len(q_tokens)))

    module_overlap = sum(1 for t in _ORACLE_MODULE_TERMS if t in dl and t in ql or t in dl and t in q_tokens)
    negative_hits = sum(1 for t in _NEGATIVE_ERP_TERMS if t in dl and t not in ql)

    legacy_in_doc = _contains_any(dl, _LEGACY_TERMS)
    fusion_in_question = _contains_any(ql, _FUSION_TERMS)
    fusion_mismatch_penalty = 1 if (fusion_in_question and legacy_in_doc and not _contains_any(ql, _LEGACY_TERMS)) else 0

    # Simple answerability check
    answerability = sum(1 for t in _ANSWERABILITY_CUES if t in dl)

    # Year signal: penalize obviously old-only docs when question implies latest/new
    years_in_doc = [int(y) for y in re.findall(r"\b(19|20)\d{2}\b", dl)]
    years_in_question = [int(y) for y in re.findall(r"\b(19|20)\d{2}\b", ql)]
    latest_in_question = ("latest" in ql or "current" in ql or (years_in_question and max(years_in_question) >= 2022))
    oldness_penalty = 1 if latest_in_question and years_in_doc and max(years_in_doc) < 2018 else 0

    # Scoring: emphasize module alignment and lexical grounding, penalize vendor/temporal mismatches
    score = (
        2.0 * module_overlap +
        8.0 * overlap_ratio +
        1.5 * min(answerability, 3) -
        2.5 * negative_hits -
        2.0 * fusion_mismatch_penalty -
        1.0 * oldness_penalty
    )

    return {
        "score": score,
        "overlap_ratio": overlap_ratio,
        "module_overlap": module_overlap,
        "negative_hits": negative_hits,
        "answerability": answerability,
        "fusion_mismatch": fusion_mismatch_penalty,
        "oldness_penalty": oldness_penalty,
    }

def _dedupe_docs(docs: List[Document]) -> List[Document]:
    """2_query.py'deki gelişmiş deduplication mantığı"""
    seen_combinations = set()
    unique_docs = []
    
    for doc in docs:
        # Parent_id + header_hierarchy + content preview kombinasyonu
        parent_id = doc.metadata.get('parent_id', 'unknown')
        header_hierarchy = doc.metadata.get('header_hierarchy', 'unknown')
        content_preview = doc.page_content[:100]
        
        combination = f"{parent_id}|{header_hierarchy}|{content_preview}"
        
        if combination not in seen_combinations:
            seen_combinations.add(combination)
            unique_docs.append(doc)
            if ENABLE_DEBUG_INFO:
                print(f"🔍 Yeni benzersiz chunk: Parent={parent_id[:8]}... | Header={header_hierarchy[:30]}...")
        else:
            if ENABLE_DEBUG_INFO:
                print(f"🔄 Yinelenen chunk kaldırıldı: Parent={parent_id[:8]}... | Header={header_hierarchy[:30]}...")
    
    if ENABLE_DEBUG_INFO:
        print(f"🧹 Deduplication sonrası: {len(docs)} -> {len(unique_docs)} chunk")
    
    return unique_docs

# 3. RAG GENERATION CHAIN
rag_prompt = PromptTemplate(
    template="""Using the CONTEXT text below, answer the given QUESTION. Summarize your answer in a paragraph using only the information from the context.

CONTEXT:
---
{context}
---

QUESTION: {question}

ANSWER:""",
    input_variables=["context", "question"],
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = rag_prompt | llm_text | StrOutputParser()

# 4. HALLUCINATION GRADER - Advanced Oracle SCM Evaluation System
hallucination_grader_prompt = PromptTemplate(
    template="""You are one of three independent Oracle SCM graders in a majority-vote system that detects hallucinations.

Your ONLY task is to decide if the Generated Response is grounded in the Given Documents. Ignore any external knowledge unless a Web Verification snippet is provided at the end. Do not assume facts not present in the sources.

Decision rules:
- Return {{"score": "yes"}} ONLY if the core factual claims in the response are directly supported by the sources (same concepts, entities, data, or procedures).
- Return {{"score": "no"}} if important claims are missing from the sources, contradict them, or appear speculative.

Oracle-specific checks to consider (when present):
- Module names and terminology (INV, BOM, WIP, MRP, Procurement, etc.)
- Forms, reports, tables, APIs, concurrent programs
- Process steps and configuration parameters
- Data names, codes, and values appearing in sources

Sources (Vectorstore Documents):
{documents}

Optional Web Verification (may be empty):
{web_verification}

Generated Response:
{generation}

Respond ONLY in JSON as one of: {{"score": "yes"}} or {{"score": "no"}}. Do not include any other fields or text.""",
    input_variables=["generation", "documents", "web_verification"],
)

hallucination_grader = hallucination_grader_prompt | llm_json | JsonOutputParser()

# Voting-based hallucination check using 3 models in parallel
def vote_hallucination(documents: List[Document], generation: str) -> Dict[str, Any]:
    content = format_docs(documents)
    votes = 0
    results = []
    for model in HALLUCINATION_MODELS:
        try:
            grader = hallucination_grader_prompt | model | JsonOutputParser()
            res = grader.invoke({"documents": content, "generation": generation, "web_verification": ""})
            vote = 1 if res.get("score", "no") == "yes" else 0
            votes += vote
            results.append(res)
        except Exception as e:
            results.append({"error": str(e)})
    majority = votes >= 2
    print(f"---🗳️ HALLUCINATION VOTES: {votes}/3 approve---")
    return {"votes": votes, "majority": majority, "raw": results}

# Web verification utilities
ORACLE_TERMS_FOR_QUERY = {
    "oracle", "manufacturing", "inventory", "inv", "bom", "routing", "wip", "mrp",
    "mps", "procurement", "purchasing", "po", "supplier", "quality", "costing",
    "wms", "warehouse", "receiving", "shipping", "planning", "concurrent program", "api"
}

def extract_web_verification_queries(question: str, generation: str, max_queries: int = 3) -> List[str]:
    text = generation.strip()
    # Simple sentence split
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # Prefer sentences mentioning Oracle terms
    prioritized = [s for s in sentences if any(term in s.lower() for term in ORACLE_TERMS_FOR_QUERY)]
    remaining = [s for s in sentences if s not in prioritized]
    candidates = prioritized + remaining
    # Ensure fallback to question if generation is too short
    queries = []
    for s in candidates:
        s_clean = s.strip()
        if len(s_clean) > 0:
            queries.append(s_clean[:180])
        if len(queries) >= max_queries:
            break
    if not queries:
        queries = [question[:180]]
    return queries

def fetch_web_verification_text(queries: List[str]) -> str:
    # Web verification disabled - vectorstore only system
    return ""

def vote_hallucination_with_web(documents: List[Document], generation: str, web_text: str) -> Dict[str, Any]:
    votes = 0
    for model in HALLUCINATION_MODELS:
        try:
            grader = hallucination_grader_prompt | model | JsonOutputParser()
            res = grader.invoke({
                "documents": format_docs(documents),
                "generation": generation,
                "web_verification": web_text,
            })
            votes += 1 if res.get("score", "no") == "yes" else 0
        except Exception:
            pass
    return {"votes": votes, "majority": votes >= 2}

# 5. TWO-LAYER ANSWER QUALITY EVALUATION SYSTEM

# LAYER 1: Oracle SCM Consultant Standards Evaluation
oracle_consultant_evaluation_prompt = PromptTemplate(
    template="""<oracle_scm_evaluation_salt_abc123>
<role>You are a Senior Oracle SCM Consultant with 10+ years of experience evaluating technical responses for Oracle Supply Chain Management solutions.</role>

<evaluation_framework>
<criteria_definition>
Oracle SCM Consultant standards require responses to demonstrate:
- Technical accuracy aligned with Oracle SCM modules and best practices
- Actionable implementation guidance for Oracle professionals  
- Comprehensive coverage appropriate to the Oracle SCM business context
- Clear articulation of Oracle-specific processes, terminology, and procedures
- Practical value for real-world Oracle SCM scenarios and challenges
</criteria_definition>

<scoring_dimensions>
<dimension name="technical_accuracy">
<description>Does the response provide technically correct Oracle SCM information?</description>
<excellent>Uses correct Oracle terminology, module names, processes, and technical details</excellent>
<acceptable>Generally accurate with minor technical gaps or simplifications</acceptable>
<insufficient>Contains technical errors, wrong terminology, or misleading information</insufficient>
</dimension>

<dimension name="oracle_specificity">
<description>How well does the response address Oracle SCM context specifically?</description>
<excellent>Clearly Oracle-focused with specific module references, forms, reports, or APIs</excellent>
<acceptable>Oracle context is present but could be more specific to SCM modules</acceptable>
<insufficient>Generic ERP information without clear Oracle SCM relevance</insufficient>
</dimension>

<dimension name="actionability">
<description>Can Oracle professionals implement or use this guidance practically?</description>
<excellent>Provides clear steps, navigation paths, or implementation details</excellent>
<acceptable>Offers useful guidance but may lack some implementation specifics</acceptable>
<insufficient>Too vague or theoretical to be practically useful</insufficient>
</dimension>

<dimension name="completeness">
<description>Does the response adequately cover the scope of the question?</description>
<excellent>Comprehensive coverage addressing all key aspects of the question</excellent>
<acceptable>Covers main points but may miss some secondary aspects</acceptable>
<insufficient>Incomplete or significantly gaps in addressing the question</insufficient>
</dimension>
</scoring_dimensions>
</evaluation_framework>

<evaluation_process>
<question_context>
<user_question>{question}</user_question>
<response_to_evaluate>{generation}</response_to_evaluate>
</question_context>

<assessment_steps>
<step number="1">Analyze the user question to understand Oracle SCM context and requirements</step>
<step number="2">Evaluate response against each scoring dimension using the criteria</step>
<step number="3">Determine if response meets Oracle SCM Consultant standards overall</step>
<step number="4">Provide final pass/fail decision with brief reasoning</step>
</assessment_steps>

<decision_criteria>
**PASS (score: "yes")**: Response meets Oracle SCM Consultant standards in at least 3 out of 4 dimensions at "acceptable" level or higher, with no "insufficient" ratings in technical_accuracy or oracle_specificity.

**FAIL (score: "no")**: Response has "insufficient" ratings in technical_accuracy or oracle_specificity, OR fails to meet acceptable standards in 2 or more dimensions.
</decision_criteria>
</evaluation_process>

<output_format>
Respond only in JSON format: {{"score": "yes", "reasoning": "brief explanation"}} or {{"score": "no", "reasoning": "brief explanation"}}
</output_format>
</oracle_scm_evaluation_salt_abc123>""",
    input_variables=["generation", "question"],
)

# LAYER 2: Unbiased LLM Quality Assessment
unbiased_quality_evaluation_prompt = PromptTemplate(
    template="""<unbiased_quality_check_def456>
<role>You are an independent quality assessor evaluating answer quality without domain-specific bias. You have no knowledge of the previous evaluation.</role>

<quality_assessment_framework>
<evaluation_principles>
<principle>Assess answer quality based purely on general communication and information standards</principle>
<principle>Evaluate without considering specialized domain knowledge or terminology</principle>
<principle>Focus on clarity, coherence, relevance to the question, and information completeness</principle>
<principle>Do not favor technical jargon or domain-specific language over clear explanations</principle>
</evaluation_principles>

<quality_dimensions>
<dimension name="relevance">
<description>How well does the answer address what was asked?</description>
<criteria>Answer should directly respond to the core question and stay on topic</criteria>
</dimension>

<dimension name="clarity">
<description>Is the answer clear and understandable?</description>
<criteria>Information should be presented in a logical, easy-to-follow manner</criteria>
</dimension>

<dimension name="completeness">
<description>Does the answer provide sufficient information?</description>
<criteria>Answer should adequately cover the question scope without major gaps</criteria>
</dimension>

<dimension name="coherence">
<description>Is the answer logically structured and consistent?</description>
<criteria>Information should flow logically and not contain contradictions</criteria>
</dimension>
</quality_dimensions>
</quality_assessment_framework>

<assessment_task>
<input_question>{question}</input_question>
<answer_to_assess>{generation}</answer_to_assess>

<evaluation_instructions>
<instruction>Read the question and answer independently</instruction>
<instruction>Assess each quality dimension based on general communication standards</instruction>
<instruction>Determine if the answer meets basic quality expectations for a professional response</instruction>
<instruction>Make a binary decision: acceptable quality or needs improvement</instruction>
</evaluation_instructions>

<decision_criteria>
**ACCEPTABLE (score: "yes")**: Answer is relevant, clear, reasonably complete, and coherent. A professional would find this response satisfactory as an initial answer to their question.

**NEEDS_IMPROVEMENT (score: "no")**: Answer has significant issues with relevance, clarity, completeness, or coherence that would make it unsatisfactory for professional use.
</decision_criteria>
</assessment_task>

<output_requirement>
Respond only in JSON format: {{"score": "yes", "assessment": "brief quality summary"}} or {{"score": "no", "assessment": "brief quality summary"}}
</output_requirement>
</unbiased_quality_check_def456>""",
    input_variables=["generation", "question"],
)

# Create evaluation chains
oracle_consultant_evaluator = oracle_consultant_evaluation_prompt | llm_json | JsonOutputParser()
unbiased_quality_evaluator = unbiased_quality_evaluation_prompt | llm_json | JsonOutputParser()

# 6. QUESTION REWRITER - Conservative Oracle SCM Query Enhancement
rewrite_prompt = PromptTemplate(
    template="""You are an Oracle SCM search optimization expert. Your task is to MINIMALLY enhance the user question while preserving its original meaning and structure.

<conservative_enhancement_principles>
- PRESERVE the original question structure and intent
- ADD only essential Oracle terminology that might appear in documentation
- DO NOT change the core meaning or scope of the question
- Keep enhancements subtle and relevant
- Maintain natural language flow
</conservative_enhancement_principles>

<minimal_enhancement_strategy>
**Step 1 - Identify Core Intent:**
Understand what the user is specifically asking for

**Step 2 - Minimal Oracle Context:**
Add only the most relevant Oracle module or terminology if missing

**Step 3 - Preserve Structure:**
Keep the original question format and flow intact

**Step 4 - Conservative Addition:**
Add 1-2 essential terms that would improve document matching
</minimal_enhancement_strategy>

<enhancement_examples>
**Example 1 - Manufacturing Focus:**
Original: "How to create production order"
Enhanced: "How to create production order in Oracle Manufacturing"
Reasoning: Added module context without changing structure

**Example 2 - Technical Process:**
Original: "Setup routing operations"
Enhanced: "Setup routing operations work center"
Reasoning: Added related term that appears in Oracle docs

**Example 3 - Already Specific:**
Original: "How to configure Oracle WIP discrete job parameters"
Enhanced: "How to configure Oracle WIP discrete job parameters"
Reasoning: No change needed - already well-specified

**Example 4 - General Business:**
Original: "Inventory management best practices"
Enhanced: "Oracle inventory management best practices"
Reasoning: Added Oracle context for better targeting
</enhancement_examples>

<rewrite_guidelines>
**DO:**
- Add missing "Oracle" if not present and relevant
- Include relevant module name (Manufacturing, WIP, INV) if clearly implied
- Add one closely related technical term if it enhances search
- Keep the question readable and natural

**DON'T:**
- Completely restructure the question
- Add multiple unrelated terms
- Change the question scope or intent
- Make the question overly technical if original was simple
- Add domain-specific jargon unless clearly relevant
</rewrite_guidelines>

<enhancement_execution>
Original Question: {question}

**Analysis:**
- Core intent: [What is the user asking?]
- Current Oracle context: [Is Oracle context already present?]
- Missing key terms: [What 1-2 terms would help document matching?]

**Conservative Enhancement:**
[Provide minimally enhanced question that preserves meaning while improving search effectiveness]
</enhancement_execution>

Provide only the enhanced question without explanations.""",
    input_variables=["question"],
)

question_rewriter = rewrite_prompt | llm_text | StrOutputParser()

# =============================================================================
# CHROME BROWSER MCP WEB SEARCH SETUP
# =============================================================================

"""
🌐 CHROME BROWSER MCP WEB SEARCH EXPLANATION:

The web_search node now uses Chrome Browser via Model Context Protocol (MCP) for real web browsing.

📋 How it works:
1. MultiServerMCPClient connects to @modelcontextprotocol/server-puppeteer
2. Chrome browser is launched and controlled via Puppeteer
3. Real web pages are navigated and content is extracted
4. Multiple search strategies: Google search + Oracle documentation
5. Content is cleaned and processed for RAG pipeline
6. Fallback to Tavily if Chrome MCP fails

🔄 Process Flow:
Question → Chrome MCP → Real Browser → Google/Oracle Search → Content Extract → Document Object → RAG Generation

⚡ Performance:
- Real browser rendering and JavaScript execution
- Access to dynamic content
- Multiple search sources
- Structured content extraction

🛡️ Fallback:
- If Chrome MCP fails, falls back to Tavily
- If both fail, uses vectorstore retrieval
- Comprehensive error handling

This provides more comprehensive and current information through real web browsing.
"""

# =============================================================================
# WEB SEARCH SETUP
# =============================================================================


try:
    load_dotenv()
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    if tavily_api_key:
        web_search_tool = TavilySearchResults(k=3)
        print("✅ Web search tool activated.")
        WEB_SEARCH_ENABLED = True
    else:
        print("⚠️ TAVILY_API_KEY not found. Web search disabled.")
        web_search_tool = None
        WEB_SEARCH_ENABLED = False
        
except Exception as e:
    print(f"⚠️ Web search setup error: {e}")
    web_search_tool = None
    WEB_SEARCH_ENABLED = False



# =============================================================================
# SIMPLE WEB SEARCH - DeepAgents removed
# =============================================================================

# =============================================================================
# LANGGRAPH STATE AND NODE FUNCTIONS
# =============================================================================

class GraphState(TypedDict):
    """Graph state - only question visible to user, rest are internal system state"""
    # User input (required field)
    question: str
    
    # Internal system state (all optional with NotRequired)
    generation: NotRequired[str]
    documents: NotRequired[List[str]]
    generation_attempts: NotRequired[int]
    hallucination_failures: NotRequired[int]
    query_rewrites: NotRequired[int]
    route_source: NotRequired[str]
    previous_generation: NotRequired[str]
    hallucination_votes: NotRequired[int]
    hallucination_majority: NotRequired[bool]
    web_verification_attempted: NotRequired[bool]
    
    # Document retrieval retry tracking
    retrieval_attempts: NotRequired[int]
    excluded_doc_ids: NotRequired[List[str]]
    grading_decision: NotRequired[str]
    
    # Hallucination routing decision
    hallucination_decision: NotRequired[str]
    
    # Advanced Retrieval Pipeline Info (for LangGraph Studio visibility)
    pipeline_info: NotRequired[Dict[str, Any]]
    document_count: NotRequired[int]
    retrieval_method: NotRequired[str]
    pipeline_summary: NotRequired[str]
    generated_queries: NotRequired[List[str]]
    parent_child_mapping: NotRequired[Dict[str, Any]]
    



### STATE INITIALIZATION HELPERS ###

def initialize_state_defaults(state: dict) -> dict:
    """Initialize all state fields with defaults for UI visibility"""
    defaults = {
        "generation": "",
        "documents": [],
        "generation_attempts": 0,
        "hallucination_failures": 0,
        "query_rewrites": 0,
        "route_source": "",
        "previous_generation": "",
        "hallucination_votes": 0,
        "hallucination_majority": False,
        "web_verification_attempted": False,
        # Document retrieval retry tracking
        "retrieval_attempts": 0,
        "excluded_doc_ids": [],
        "grading_decision": "",
        # Hallucination routing decision
        "hallucination_decision": "continue",
        # Two-layer quality evaluation fields
        "oracle_evaluation_score": "",
        "oracle_evaluation_reasoning": "",
        "unbiased_evaluation_score": "",
        "unbiased_evaluation_assessment": "",
        "quality_consensus": "",
        "quality_bypass": False,
        # Advanced Retrieval Pipeline Info
        "pipeline_info": {},
        "document_count": 0,
        "retrieval_method": "",
        "pipeline_summary": "",
        "generated_queries": [],
        "parent_child_mapping": {},

    }
    
    # Apply defaults for missing keys
    for key, default_value in defaults.items():
        if key not in state:
            state[key] = default_value
    
    return state

### NODE FUNCTIONS ###

def retrieve(state):
    """Advanced retrieve with MultiQuery + Cross-Encoder + Parent Mapping (from 2_query.py)"""
    print("---🔥 RETRIEVE: MultiQuery+CrossEncoder+ParentMapping PIPELINE---")
    state = initialize_state_defaults(state)
    question = state["question"]
    excluded_doc_ids = state["excluded_doc_ids"]
    retrieval_attempts = state["retrieval_attempts"]
    
    # Increment retrieval attempts counter
    retrieval_attempts += 1
    print(f"---📊 RETRIEVAL ATTEMPT: {retrieval_attempts}---")
    
    try:
        if ENABLE_DEBUG_INFO:
            print(f"🔍 Soru: {question}")
            print("📋 ADVANCED RAG PIPELINE BAŞLATIYOR...")
            
            # Pipeline bileşenlerini detaylandır
            pipeline_components = []
            if hasattr(retriever, 'base_retriever') and 'MultiQuery' in type(retriever.base_retriever).__name__:
                pipeline_components.append("✅ MultiQuery (5 alternatif sorgu)")
            if hasattr(retriever, 'base_compressor'):
                pipeline_components.append("✅ Cross-Encoder (BAAI/bge-reranker-base)")
            if parent_store:
                pipeline_components.append(f"✅ Parent Mapping ({len(parent_store._store)} documents)")
            
            print("🔧 PIPELINE COMPONENTS:")
            for component in pipeline_components:
                print(f"   {component}")
        
        # Ana retrieval (gelişmiş pipeline ile)
        if ENABLE_DEBUG_INFO:
            print("🚀 MultiQuery + Cross-Encoder + Parent Mapping başlatılıyor...")
        
        # MultiQuery sorularını yakala ve göster
        generated_queries = []
        multiquery_info = ""
        
        if hasattr(retriever, 'base_retriever') and 'MultiQuery' in type(retriever.base_retriever).__name__:
            try:
                # MultiQuery'nin doğrudan generate_queries metodunu kullan
                multiquery_retriever = retriever.base_retriever
                generated_queries = multiquery_retriever.generate_queries(question)
                
                multiquery_info = f"MultiQuery: {len(generated_queries)} sorgu üretildi"
                if ENABLE_DEBUG_INFO:
                    print(f"🔍 Generated queries:")
                    for i, q in enumerate(generated_queries, 1):
                        print(f"   {i}. {q}")
            except Exception as e:
                if ENABLE_DEBUG_INFO:
                    print(f"⚠️ MultiQuery generation error: {e}")
                # Fallback: basit sorgu çeşitlemeleri
                generated_queries = [
                    question,
                    f"Oracle {question}",
                    f"How to {question}",
                    f"{question} best practices",
                    f"{question} Oracle ERP"
                ]
                multiquery_info = f"MultiQuery: {len(generated_queries)} fallback sorgu"
        
        documents = retriever.invoke(question)
        
        # Deduplication (2_query.py'deki gibi)
        if len(documents) > 0:
            if ENABLE_DEBUG_INFO:
                print(f"\n🧹 DEDUPLICATION İŞLEMİ")
            documents = _dedupe_docs(documents)
        
        # Cross-encoder bilgilerini yakala
        crossencoder_info = ""
        if hasattr(retriever, 'base_compressor'):
            crossencoder_info = f"CrossEncoder: {retriever.base_compressor.model.model_name}"
        
        if ENABLE_DEBUG_INFO:
            print(f"📄 {len(documents)} child document bulundu")
        
        # Filter out previously excluded documents if any
        if excluded_doc_ids:
            for i, doc in enumerate(documents):
                if not hasattr(doc, "metadata") or not doc.metadata.get("id"):
                    doc_id = f"doc_{hash(doc.page_content[:100])}"
                    if not hasattr(doc, "metadata"):
                        doc.metadata = {}
                    doc.metadata["id"] = doc_id
            
            original_count = len(documents)
            documents = [doc for doc in documents if doc.metadata.get("id") not in excluded_doc_ids]
            if ENABLE_DEBUG_INFO:
                print(f"📄 {len(documents)} documents after filtering ({original_count - len(documents)} excluded)")
        
        # DIVERSITY FACTOR: Parent çeşitliliğini artır (2_query.py EXACT mantığı)
        if DIVERSITY_FACTOR and len(documents) > 0:
            if ENABLE_DEBUG_INFO:
                print(f"\n🌐 ÇEŞİTLİLİK FAKTÖRÜ UYGULANACAK")
            # Parent ID'lere göre grupla
            parent_groups = {}
            for doc in documents:
                parent_id = doc.metadata.get('parent_id', 'unknown')
                if parent_id not in parent_groups:
                    parent_groups[parent_id] = []
                parent_groups[parent_id].append(doc)
            
            if ENABLE_DEBUG_INFO:
                print(f"📊 {len(parent_groups)} farklı parent'tan chunk'lar bulundu")
            
            # Her parent'tan en iyi chunk'ı seç, çeşitliliği artır
            diverse_docs = []
            for parent_id, docs in parent_groups.items():
                # En yüksek skorlu chunk'ı seç
                best_doc = max(docs, key=lambda x: x.metadata.get('relevance_score', 0.0))
                diverse_docs.append(best_doc)
                if ENABLE_DEBUG_INFO:
                    print(f"   🔍 Parent {parent_id[:8]}...: En iyi chunk seçildi (skor: {best_doc.metadata.get('relevance_score', 0.0):.4f})")
            
            # Çeşitlilik sonrası durumu göster
            if ENABLE_DEBUG_INFO:
                print(f"   ✅ Çeşitlilik sonrası: {len(documents)} -> {len(diverse_docs)} chunk")
            documents = diverse_docs

        # Cross-encoder puanlarını metadata'ya ekle (2_query.py mantığı)
        if ENABLE_CROSS_ENCODER and hasattr(retriever, 'base_compressor'):
            try:
                # Reranking puanlarını yeniden hesapla (tüm birleşik liste üzerinde)
                pairs = [(question, doc.page_content) for doc in documents]
                if pairs:
                    scores = retriever.base_compressor.model.predict(pairs)
                    for doc, score in zip(documents, scores):
                        doc.metadata['relevance_score'] = float(score)
                    if ENABLE_DEBUG_INFO:
                        print("✅ Cross-encoder reranking puanları metadata'ya eklendi")
                
                # Eğer çeşitlilik ile listeyi genişlettiysek, Top-N'e kırp
                try:
                    top_n = int(getattr(retriever.base_compressor, 'top_n', 0) or 0)
                except Exception:
                    top_n = 0
                if top_n and len(documents) > top_n:
                    documents = sorted(documents, key=lambda d: d.metadata.get('relevance_score', 0.0), reverse=True)[:top_n]
                    if ENABLE_DEBUG_INFO:
                        print(f"✂️ Reranking sonrası en iyi {top_n} chunk seçildi")
            except Exception as e:
                if ENABLE_DEBUG_INFO:
                    print(f"⚠️ Reranking puanları hesaplanamadı: {e}")
                # Fallback: sıralama indeksine göre azalan puanlar ver
                for i, doc in enumerate(documents):
                    doc.metadata['relevance_score'] = 1.0 - (i * 0.1)
        else:
            # Fallback scoring for non-cross-encoder setups
            for i, doc in enumerate(documents):
                doc.metadata['relevance_score'] = 0.8 - (i * 0.05)
        
        # Parent document mapping (2_query.py'deki ana özellik)
        if parent_store and documents:
            if ENABLE_DEBUG_INFO:
                print("🏗️ Parent document mapping başlatılıyor...")
            result = get_final_parent_documents(documents)
            documents = result['final_docs']
            
            # Çeşitlilik faktörü uygula
            if DIVERSITY_FACTOR and len(documents) > RERANK_TOP_N:
                documents = documents[:RERANK_TOP_N]
                if ENABLE_DEBUG_INFO:
                    print(f"✂️ Çeşitlilik sonrası en iyi {RERANK_TOP_N} parent seçildi")
        
        if ENABLE_DEBUG_INFO and documents:
            print(f"\n✅ FINAL PIPELINE RESULT: {len(documents)} documents selected")
            print("📊 FINAL DOCUMENTS WITH PIPELINE INFO:")
            for i, doc in enumerate(documents, 1):
                score = doc.metadata.get('relevance_score', 0.0)
                hierarchy = doc.metadata.get('header_hierarchy', 'N/A')[:50]
                content_len = len(doc.page_content)
                parent_id = doc.metadata.get('parent_id', 'NO_PARENT')
                source = doc.metadata.get('source', 'unknown')
                print(f"   {i}. [Score:{score:.4f}] [ParentID:{parent_id}] {hierarchy}")
                print(f"      Source: {source} | Length: {content_len} chars")
            
            print(f"\n🎯 PIPELINE SUMMARY:")
            print(f"   MultiQuery: {multiquery_info}")
            print(f"   CrossEncoder: {crossencoder_info}")
            print(f"   Generated Queries: {len(generated_queries)} sorgu")
            print(f"   Parent Mapping: {len(pipeline_info['parent_child_mapping'])} unique parents")
        
        # Update state with documents and detailed pipeline info for LangGraph Studio visibility
        pipeline_info = {
            "multiquery_enabled": ENABLE_MULTIQUERY,
            "cross_encoder_enabled": ENABLE_CROSS_ENCODER,
            "parent_mapping_enabled": bool(parent_store),
            "retriever_type": type(retriever).__name__,
            "pipeline_stages": [],
            "multiquery_details": multiquery_info,
            "crossencoder_details": crossencoder_info,
            "generated_queries": generated_queries,
            "documents_with_parent_id": sum(1 for doc in documents if doc.metadata.get('parent_id')),
            "documents_without_parent_id": sum(1 for doc in documents if not doc.metadata.get('parent_id')),
            "parent_child_mapping": {}
        }
        
        # Parent-Child mapping detayları ve debugging
        if ENABLE_DEBUG_INFO:
            print(f"\n🔍 PARENT-CHILD MAPPING ANALYSIS:")
            
        for i, doc in enumerate(documents):
            parent_id = doc.metadata.get('parent_id')
            if ENABLE_DEBUG_INFO:
                print(f"   📄 Doc {i+1}: parent_id = {parent_id if parent_id else 'MISSING'}")
                print(f"       Metadata keys: {list(doc.metadata.keys())}")
            
            # Parent ID eksikse no_parent ID ata
            if not parent_id:
                parent_id = f'no_parent_{i}'
                if ENABLE_DEBUG_INFO:
                    print(f"       ⚠️ No parent_id found! Assigned: {parent_id}")
            
            if parent_id not in pipeline_info["parent_child_mapping"]:
                pipeline_info["parent_child_mapping"][parent_id] = {
                    "parent_content_preview": doc.page_content[:100] + "...",
                    "relevance_score": doc.metadata.get('relevance_score', 0.0),
                    "source": doc.metadata.get('source', 'unknown'),
                    "header_hierarchy": doc.metadata.get('header_hierarchy', 'N/A'),
                    "content_length": len(doc.page_content),
                    "chunk_diversity": doc.metadata.get('chunk_diversity', 1),
                    "total_chunks_from_parent": doc.metadata.get('total_chunks_from_parent', 1),
                    "has_real_parent_id": bool(doc.metadata.get('parent_id'))
                }
        
        # Pipeline stage tracking
        if hasattr(retriever, 'base_retriever'):
            pipeline_info["pipeline_stages"].append(f"MultiQuery({type(retriever.base_retriever).__name__})")
        if hasattr(retriever, 'base_compressor'):
            pipeline_info["pipeline_stages"].append(f"CrossEncoder({retriever.base_compressor.model.model_name})")
        if parent_store:
            pipeline_info["pipeline_stages"].append(f"ParentMapping({len(parent_store._store)}docs)")
        
        state.update({
            "documents": documents,
            "route_source": "advanced_chroma_vectorstore",
            "retrieval_attempts": retrieval_attempts,
            "pipeline_info": pipeline_info,
            "document_count": len(documents),
            "retrieval_method": "→".join(pipeline_info["pipeline_stages"]),
            "pipeline_summary": f"{multiquery_info} | {crossencoder_info} | ParentMapping: {len(documents)} docs",
            "generated_queries": generated_queries,
            "parent_child_mapping": pipeline_info["parent_child_mapping"]
        })
        return state
        
    except Exception as e:
        print(f"⚠️ Advanced retrieve error: {e}")
        # Fallback to simple retrieval
        try:
            simple_docs = vectorstore.similarity_search(question, k=5)
            for i, doc in enumerate(simple_docs):
                doc.metadata['relevance_score'] = 0.6 - (i * 0.1)
            print(f"🔄 Fallback retrieval: {len(simple_docs)} documents")
        except Exception as fallback_error:
            print(f"❌ Fallback retrieval failed: {fallback_error}")
            simple_docs = []
            
        state.update({
            "documents": simple_docs,
            "route_source": "fallback_vectorstore",
            "retrieval_attempts": retrieval_attempts
        })
        return state
def generate(state):
    """Generate answer using RAG with attempt tracking"""
    print("---✨ GENERATE---")
    state = initialize_state_defaults(state)
    question = state["question"]
    documents = state["documents"]
    
    # Track generation attempts
    generation_attempts = state["generation_attempts"] + 1
    print(f"---📊 GENERATION ATTEMPT: {generation_attempts}---")
    
    # Debug: Show context
    print("\n" + "="*20 + " CONTEXT COMING TO GENERATE NODE " + "="*20)
    context_for_llm = format_docs(documents)
    print(context_for_llm[:500] + "..." if len(context_for_llm) > 500 else context_for_llm)
    print("="*65 + "\n")

    if not documents:
        print("⚠️ No documents, general answer provided...")
        generation = "I'm sorry, I don't have enough information on this topic from our Oracle documentation. This might require current information from web sources."
    else:
        try:
            generation = rag_chain.invoke({"context": context_for_llm, "question": question})
        except Exception as e:
            print(f"⚠️ Generation error: {e}")
            generation = "An error occurred while generating the answer."
    
    # Update state with generation results
    state.update({
        "generation": generation,
        "generation_attempts": generation_attempts,
        "previous_generation": state.get("generation", "")  # Store current as previous for next iteration
    })
    return state

def grade_documents(state):
    """Advanced document grading with 2_query.py heuristics and parent mapping"""
    print("---📋 ADVANCED GRADE DOCUMENTS---")
    state = initialize_state_defaults(state)
    question = state["question"]
    documents = state["documents"]
    retrieval_attempts = state["retrieval_attempts"]
    excluded_doc_ids = state["excluded_doc_ids"]
    
    if not documents:
        print("---⚠️ NO DOCUMENTS---")
        if retrieval_attempts >= 3:
            print("---🚨 THREE RETRIEVAL ATTEMPTS WITH NO DOCUMENTS, USING WEB SEARCH---")
            state["grading_decision"] = "use_web_search"
        elif retrieval_attempts == 2:
            print("---🔄 SECOND RETRIEVAL ATTEMPT WITH NO DOCUMENTS, TRYING QUERY REWRITE---")
            state["grading_decision"] = "transform_query"
        else:
            print("---🔄 FIRST RETRIEVAL ATTEMPT WITH NO DOCUMENTS, TRYING AGAIN---")
            state["grading_decision"] = "retry_retrieval"
        return state
    
    # Advanced grading with 2_query.py logic
    candidates: List[Dict[str, Any]] = []
    max_docs_to_check = min(RERANK_TOP_N, len(documents))
    HIGH_THRESHOLD = 6.0
    LOW_THRESHOLD = 3.0
    current_doc_ids = []
    
    if ENABLE_DEBUG_INFO:
        print(f"📊 Grading {max_docs_to_check} documents with advanced heuristics...")
    
    for i, d in enumerate(documents[:max_docs_to_check]):
        # Ensure document has an ID for tracking
        if not hasattr(d, "metadata") or not d.metadata.get("id"):
            doc_id = f"doc_{hash(d.page_content[:100])}"
            if not hasattr(d, "metadata"):
                d.metadata = {}
            d.metadata["id"] = doc_id
        
        current_doc_ids.append(d.metadata.get("id"))
        doc_content = d.page_content[:2000]
        
        try:
            # 2_query.py'deki heuristic scoring
            signals = heuristic_relevance_signals(question, doc_content)
            
            if ENABLE_DEBUG_INFO:
                print(f"📈 DOC {i+1}: score={signals['score']:.2f}, overlap={signals['overlap_ratio']:.2f}, "
                      f"modules={signals['module_overlap']}, neg={signals['negative_hits']}")

            # Negative ERP terms check (2_query.py mantığı)
            if signals["negative_hits"] >= 1 and signals["module_overlap"] == 0:
                if ENABLE_DEBUG_INFO:
                    print(f"⛔ DOC {i+1} rejected (non-Oracle ERP noise)")
                continue

            # High-confidence accept (2_query.py mantığı)
            if signals["score"] >= HIGH_THRESHOLD:
                if ENABLE_DEBUG_INFO:
                    print(f"✅ DOC {i+1} accepted (high heuristic score)")
                candidates.append({"doc": d, "score": signals["score"], "signals": signals})
                continue

            # LLM grading for ambiguous cases
            try:
                grade_output = retrieval_grader.invoke({"question": question, "document": doc_content})
                grade = grade_output.get("score", "no")

                if grade == "yes" and signals["score"] >= LOW_THRESHOLD:
                    if ENABLE_DEBUG_INFO:
                        print(f"✅ DOC {i+1} accepted (LLM yes + heuristic ok)")
                    candidates.append({"doc": d, "score": signals["score"], "signals": signals})
                else:
                    if ENABLE_DEBUG_INFO:
                        print(f"❌ DOC {i+1} rejected (LLM={grade}, score={signals['score']:.2f})")
            except Exception as llm_error:
                if ENABLE_DEBUG_INFO:
                    print(f"⚠️ DOC {i+1} LLM grading error: {llm_error}")
                # Fallback to heuristic only
                if signals["score"] >= LOW_THRESHOLD:
                    candidates.append({"doc": d, "score": signals["score"], "signals": signals})

        except Exception as e:
            if ENABLE_DEBUG_INFO:
                print(f"⚠️ DOC {i+1} grading error: {e} -> using fallback")
            
            # Enhanced fallback with oracle_grader_fallback
            fallback_score = oracle_grader_fallback(question, doc_content[:600])
            if fallback_score["score"] == "yes":
                fallback_signals = heuristic_relevance_signals(question, doc_content)
                candidates.append({"doc": d, "score": max(LOW_THRESHOLD, fallback_signals["score"]), "signals": fallback_signals})

    # Handle no passing documents (2_query.py logic)
    if not candidates and documents:
        print("---🚨 NO DOCUMENTS PASSED GRADING---")
        
        if retrieval_attempts >= 3:
            print("---🚨 THREE RETRIEVAL ATTEMPTS WITH NO PASSING DOCUMENTS, USING WEB SEARCH---")
            state["grading_decision"] = "use_web_search"
            # Keep top-2 by heuristic as fallback
            scored_all = [
                (idx, heuristic_relevance_signals(question, doc.page_content[:2000]))
                for idx, doc in enumerate(documents[:max_docs_to_check])
            ]
            scored_all.sort(key=lambda x: x[1]["score"], reverse=True)
            for idx, sig in scored_all[:2]:
                candidates.append({"doc": documents[idx], "score": sig["score"], "signals": sig})
        elif retrieval_attempts == 2:
            print("---🔄 SECOND RETRIEVAL ATTEMPT WITH NO PASSING DOCUMENTS, TRYING QUERY REWRITE---")
            state["grading_decision"] = "transform_query"
            state["excluded_doc_ids"] = excluded_doc_ids + current_doc_ids
            return state
        else:
            print("---🔄 FIRST RETRIEVAL ATTEMPT WITH NO PASSING DOCUMENTS, TRYING AGAIN---")
            state["grading_decision"] = "retry_retrieval"
            state["excluded_doc_ids"] = excluded_doc_ids + current_doc_ids
            return state
    else:
        state["grading_decision"] = "generate"

    # Final ranking and selection (2_query.py mantığı)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    filtered_docs = [c["doc"] for c in candidates]
    
    # Apply final limit
    if len(filtered_docs) > RERANK_TOP_N:
        filtered_docs = filtered_docs[:RERANK_TOP_N]

    if ENABLE_DEBUG_INFO:
        print(f"✅ {len(filtered_docs)}/{len(documents)} DOCUMENTS SELECTED")
        print("📊 Final selected documents:")
        for i, doc in enumerate(filtered_docs, 1):
            score = doc.metadata.get('relevance_score', 0.0)
            hierarchy = doc.metadata.get('header_hierarchy', 'N/A')[:40]
            print(f"   {i}. [{score:.4f}] {hierarchy}")
    
    state.update({"documents": filtered_docs})
    return state

def transform_query(state):
    """Rewrite query for vectorstore search with attempt tracking"""
    print("---🔄 TRANSFORM QUERY---")
    state = initialize_state_defaults(state)
    question = state["question"]
    documents = state["documents"]
    
    # Track query rewrite attempts
    query_rewrites = state["query_rewrites"] + 1
    print(f"---📊 QUERY REWRITE ATTEMPT: {query_rewrites}---")
    
    # If too many rewrites, try web search as fallback
    if query_rewrites > 2:
        print("---🚨 TOO MANY QUERY REWRITES, SUGGESTING WEB SEARCH FALLBACK---")
        if WEB_SEARCH_ENABLED:
            return web_search({
                **state,
                "query_rewrites": query_rewrites,
                "route_source": "fallback_web_search"
            })
        else:
            return retrieve(state)
    
    try:
        better_question = question_rewriter.invoke({"question": question})
        print(f"📝 New question: {better_question}")
        state.update({
            "question": better_question,
            "query_rewrites": query_rewrites
        })
        return state
    except Exception as e:
        print(f"⚠️ Query rewrite error: {e}")
        fallback_question = f"{question} Oracle Manufacturing ERP"
        print(f"📝 Fallback question: {fallback_question}")
        state.update({
            "question": fallback_question,
            "query_rewrites": query_rewrites
        })
        return state


def web_search(state):
    """Perform web search with Tavily"""
    print("---🌐 WEB SEARCH---")
    state = initialize_state_defaults(state)
    question = state["question"]
    
    if WEB_SEARCH_ENABLED and web_search_tool:
        try:
            docs = web_search_tool.invoke({"query": question})
            web_results = "\n".join([d["content"] for d in docs])
            web_results = Document(page_content=web_results)
            
            print("---✅ WEB SEARCH SUCCESSFUL---")
            state.update({
                "documents": [web_results],
                "route_source": "web_search"
            })
            return state
        except Exception as e:
            print(f"---⚠️ WEB SEARCH ERROR: {e}---")
            return retrieve(state)
    else:
        print("Web search disabled, using vectorstore...")
        return retrieve(state)

def web_search_fallback(state):
    """Web search fallback function"""
    print("---🌐 WEB SEARCH FALLBACK---")
    return web_search(state)

### EDGE FUNCTIONS ###

def route_question(state):
    """Route question to web search or RAG."""
    print("---ROUTE QUESTION---")
    state = initialize_state_defaults(state)  # Bu satırı ekleyin
    question = state["question"]
    
    # Get the routing decision from the LLM
    route = question_router.invoke({"question": question})
    
    # Print the reasoning for the decision
    print(f"Decision: {route.datasource}")
    print(f"Reasoning Category: {route.reasoning_category}")
    print(f"Reasoning: {route.reasoning}")
    
    if route.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif route.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "retrieve"

def decide_after_grading(state):
    """Routing decision based on document grading results with retry logic"""
    print("---🤔 DOCUMENT GRADING DECISION---")
    state = initialize_state_defaults(state)
    filtered_documents = state["documents"]
    query_rewrites = state["query_rewrites"]
    retrieval_attempts = state["retrieval_attempts"]
    grading_decision = state.get("grading_decision", "")
    
    print(f"---📊 ANALYSIS: docs={len(filtered_documents)}, rewrites={query_rewrites}, retrieval_attempts={retrieval_attempts}---")
    
    # Follow the grading decision if available
    if grading_decision == "retry_retrieval":
        print("---🔄 RETRYING RETRIEVAL WITH DIFFERENT DOCUMENTS---")
        return "retrieve"
    elif grading_decision == "transform_query":
        print("---🔄 REWRITING QUERY FOR BETTER RESULTS---")
        return "transform_query"
    elif grading_decision == "use_web_search":
        print("---🌐 USING WEB SEARCH AFTER MULTIPLE RETRIEVAL FAILURES---")
        return "web_search"
    elif grading_decision == "generate":
        print(f"---✅ DOCUMENTS PASSED GRADING, GENERATING ANSWER---")
        return "generate"
    
    # Fallback logic if no explicit decision was made
    if not filtered_documents:
        print("---🔄 NO DOCUMENTS, REWRITE QUERY---")
        return "transform_query"
    
    # Avoid infinite loops - force generation after 2 rewrites
    if query_rewrites >= 2:
        print(f"---⚠️ MAX REWRITES REACHED, FORCE GENERATION---")
        return "generate"
    
    # Have documents - proceed to generation
    print(f"---✅ HAVE {len(filtered_documents)} DOCUMENTS, GENERATE ANSWER---")
    return "generate"

def check_hallucination(state):
    """Separate node for hallucination detection with voting and web verification"""
    print("---🗳️ HALLUCINATION CHECK (VOTING)---")
    state = initialize_state_defaults(state)
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    route_source = state.get("route_source", "")
    
    # Web search'ten gelen cevapları daha toleranslı karşıla
    if "web_search" in route_source:
        print("---🌐 WEB SEARCH SOURCE DETECTED - USING RELAXED HALLUCINATION CHECK---")
        # Web search'ten gelen cevaplar zaten dış kaynaklardan geldiği için hallucination riski düşük
        state["hallucination_votes"] = 3
        state["hallucination_majority"] = True
        state["hallucination_decision"] = "continue"
        print("---✅ WEB SEARCH CONTENT APPROVED (RELAXED CHECK)---")
        return state
    
    try:
        # Normal hallucination check via majority vote from 3 models
        print("---STEP 1: VOTING-BASED HALLUCINATION CHECK---")
        vote = vote_hallucination(documents, generation)
        state["hallucination_votes"] = vote["votes"]
        state["hallucination_majority"] = vote["majority"]
        hallucination_grade = "yes" if vote["majority"] else "no"

        # Optional web verification if vote failed
        if hallucination_grade == "no" and WEB_SEARCH_ENABLED and not state.get("web_verification_attempted", False):
            print("---🌐 VOTE FAILED. TRYING ONE-TIME WEB VERIFICATION---")
            try:
                queries = extract_web_verification_queries(question, generation, max_queries=3)
                web_text = fetch_web_verification_text(queries)
                if web_text:
                    vote2 = vote_hallucination_with_web(documents, generation, web_text)
                    state["hallucination_votes"] = vote2["votes"]
                    state["hallucination_majority"] = vote2["majority"]
                    state["web_verification_attempted"] = True
                    hallucination_grade = "yes" if vote2["majority"] else "no"
                    if hallucination_grade == "yes":
                        print("---✅ PASSED AFTER WEB VERIFICATION---")
                    else:
                        print("---❌ STILL FAILING AFTER WEB VERIFICATION---")
                else:
                    print("---ℹ️ NO WEB TEXT RETRIEVED FOR VERIFICATION---")
                    state["web_verification_attempted"] = True
            except Exception as e:
                print(f"---⚠️ WEB VERIFICATION ERROR: {e}---")
        
        if hallucination_grade == "no":
            print("---❌ ANSWER CONTAINS HALLUCINATION---")
            state["hallucination_failures"] = state["hallucination_failures"] + 1
            state["previous_generation"] = generation
            
            # Add hallucination warning to the generation
            votes_info = f"{state['hallucination_votes']}/3 models"
            hallucination_warning = (
                f"\n\n[POTENTIAL HALLUCINATION WARNING: This answer received only {votes_info} "
                f"votes confirming its accuracy. The information may not be fully supported by the "
                f"provided documentation. Proceeding to web search for verification.]"
            )
            state["generation"] = generation + hallucination_warning
            state["hallucination_decision"] = "use_web_search"
        else:
            print("---✅ ANSWER IS GROUNDED IN DOCUMENTS (MAJORITY VOTE)---")
            state["hallucination_decision"] = "continue"
            
    except Exception as e:
        print(f"⚠️ Hallucination grader error: {e} - Defaulting to pass")
        state["hallucination_decision"] = "continue"
    
    return state

def decide_after_hallucination_check(state):
    """Decide next step based on hallucination check results"""
    hallucination_decision = state.get("hallucination_decision", "continue")
    
    if hallucination_decision == "use_web_search":
        print("---🌐 POTENTIAL HALLUCINATION DETECTED, ROUTING TO WEB SEARCH---")
        if WEB_SEARCH_ENABLED:
            return "web_search"
        else:
            print("   ⚠️ Web search not available, continuing to quality check")
            return "grade_answer_quality"
    else:
        print("---✅ NO HALLUCINATION DETECTED, CONTINUING TO QUALITY CHECK---")
        return "grade_answer_quality"

def grade_answer_quality(state):
    """Two-layer evaluation system for answer quality assessment"""
    print("---📝 TWO-LAYER ANSWER QUALITY EVALUATION---")
    state = initialize_state_defaults(state)
    question = state["question"]
    generation = state["generation"]
    
    try:
        # LAYER 1: Oracle SCM Consultant Standards Evaluation
        print("---🎯 LAYER 1: ORACLE SCM CONSULTANT EVALUATION---")
        oracle_evaluation = oracle_consultant_evaluator.invoke({
            "question": question, 
            "generation": generation
        })
        oracle_score = oracle_evaluation.get("score", "no")
        oracle_reasoning = oracle_evaluation.get("reasoning", "No reasoning provided")
        
        print(f"   Oracle Consultant Score: {oracle_score}")
        print(f"   Oracle Reasoning: {oracle_reasoning}")
        
        # LAYER 2: Unbiased LLM Quality Assessment
        print("---🔍 LAYER 2: UNBIASED QUALITY ASSESSMENT---")
        unbiased_evaluation = unbiased_quality_evaluator.invoke({
            "question": question,
            "generation": generation
        })
        unbiased_score = unbiased_evaluation.get("score", "no")
        unbiased_assessment = unbiased_evaluation.get("assessment", "No assessment provided")
        
        print(f"   Unbiased Quality Score: {unbiased_score}")
        print(f"   Quality Assessment: {unbiased_assessment}")
        
        # CONSENSUS MECHANISM: Both must agree for acceptance
        both_agree_acceptable = (oracle_score == "yes" and unbiased_score == "yes")
        
        # Store evaluation results in state for routing
        state.update({
            "oracle_evaluation_score": oracle_score,
            "oracle_evaluation_reasoning": oracle_reasoning,
            "unbiased_evaluation_score": unbiased_score,
            "unbiased_evaluation_assessment": unbiased_assessment,
            "quality_consensus": "acceptable" if both_agree_acceptable else "unacceptable"
        })
        
        if both_agree_acceptable:
            print("---✅ CONSENSUS: BOTH LAYERS AGREE - ANSWER IS ACCEPTABLE---")
        else:
            print("---❌ CONSENSUS: ANSWER QUALITY INSUFFICIENT - ROUTING TO WEB SEARCH---")
            print(f"   Oracle Layer: {oracle_score}, Unbiased Layer: {unbiased_score}")
        
        return state
        
    except Exception as e:
        print(f"⚠️ Two-layer evaluation error: {e} - Defaulting to acceptable")
        # Default to acceptable on error to prevent system breakdown
        state.update({
            "oracle_evaluation_score": "yes",
            "oracle_evaluation_reasoning": f"Error in evaluation: {e}",
            "unbiased_evaluation_score": "yes", 
            "unbiased_evaluation_assessment": f"Error in evaluation: {e}",
            "quality_consensus": "acceptable"
        })
        return state

def decide_after_quality_check(state):
    """Edge function to route based on two-layer quality evaluation consensus"""
    print("---🤔 QUALITY EVALUATION ROUTING DECISION---")
    quality_consensus = state.get("quality_consensus", "acceptable")
    
    if quality_consensus == "acceptable":
        print("---✅ QUALITY CONSENSUS: PROCEEDING TO FINAL DECISION---")
        return "decide_final_action"
    else:
        print("---🌐 QUALITY CONSENSUS: ROUTING TO DIRECT WEB SEARCH---")
        return "direct_web_search"

def decide_final_action(state):
    """Final decision node with loop detection"""
    print("---🎯 FINAL DECISION---")
    state = initialize_state_defaults(state)
    question = state["question"]
    generation = state["generation"]
    
    # Track attempts for aggressive loop detection
    generation_attempts = state["generation_attempts"]
    hallucination_failures = state["hallucination_failures"]
    query_rewrites = state["query_rewrites"]
    route_source = state.get("route_source", "")
    hallucination_majority = state["hallucination_majority"]
    
    print(f"---📊 ATTEMPTS: Gen={generation_attempts}, Hall={hallucination_failures}, Rewrites={query_rewrites}---")
    
    # Store decision in state
    decision = "useful"  # Default
    
    # STRICT LOOP DETECTION: Maximum 3 generation attempts
    if generation_attempts >= 3:
        print("---🚨 MAXIMUM GENERATION ATTEMPTS REACHED (3), FORCING END---")
        if WEB_SEARCH_ENABLED and route_source != "web_search":
            print("---🌐 TRYING WEB SEARCH AS FINAL ATTEMPT---")
            decision = "web_search_fallback"
        else:
            print("---⚠️ ACCEPTING CURRENT ANSWER TO PREVENT INFINITE LOOP---")
            decision = "useful"
    
    # Early termination if too many hallucination failures
    elif hallucination_failures >= 2:
        print("---🚨 TOO MANY HALLUCINATION FAILURES, SWITCHING STRATEGY---")
        if WEB_SEARCH_ENABLED and route_source != "web_search":
            print("---🌐 SWITCHING TO WEB SEARCH FOR BETTER ACCURACY---")
            decision = "web_search_fallback"
        else:
            print("---⚠️ ACCEPTING CURRENT ANSWER---")
            decision = "useful"
    
    # Insufficient answer check
    elif not generation or "üzgünüm" in generation.lower() or len(generation.strip()) < 15:
        print(f"---❌ INSUFFICIENT ANSWER, TRY AGAIN---")
        if generation_attempts >= 2:
            print("---⚠️ TOO MANY INSUFFICIENT ATTEMPTS, SWITCHING TO QUERY REWRITE---")
            decision = "not useful"
        else:
            decision = "not useful"
    
    # Check for repeated answers (prevent same generation)
    elif generation and state.get("previous_generation", "") and generation.strip() == state.get("previous_generation", "").strip():
        print("---🔄 IDENTICAL ANSWER DETECTED, FORCING QUERY REWRITE---")
        decision = "not useful"
    
    # Check hallucination result
    elif not hallucination_majority:
        print("---❌ HALLUCINATION DETECTED, FORCING REWRITE---")
        if hallucination_failures >= 2:
            print("---🚨 MULTIPLE HALLUCINATION FAILURES---")
            if WEB_SEARCH_ENABLED and route_source != "web_search":
                print("---🌐 SWITCHING TO WEB SEARCH FOR ACCURACY---")
                decision = "web_search_fallback"
            else:
                print("---⚠️ ACCEPTING RESPONSE DESPITE CONCERNS---")
                decision = "useful"
        else:
            decision = "not useful"
    
    # All checks passed
    else:
        print("---✅ ANSWER ADDRESSES THE QUESTION. FINISHED.---")
        decision = "useful"
    
    # Store the decision in state for routing
    state["decision"] = decision
    return state

def direct_web_search(state):
    """Direct answer enhancement when quality is unacceptable - no web search available"""
    print("---⚠️ DIRECT ANSWER ENHANCEMENT (NO WEB SEARCH AVAILABLE)---")
    state = initialize_state_defaults(state)
    
    # Add context about why we're enhancing
    oracle_reasoning = state.get("oracle_evaluation_reasoning", "")
    unbiased_assessment = state.get("unbiased_evaluation_assessment", "")
    
    print(f"---📋 QUALITY ISSUES IDENTIFIED---")
    print(f"   Oracle Consultant: {oracle_reasoning}")
    print(f"   Unbiased Assessment: {unbiased_assessment}")
    
    # Enhance original answer with quality context
    original_generation = state.get("generation", "")
    enhanced_generation = f"""Based on available Oracle SCM documentation:

{original_generation}

[Note: This answer has been flagged for potential quality improvements. For the most current Oracle SCM guidance, please consult official Oracle documentation or contact Oracle support.]"""
    
    state.update({
        "generation": enhanced_generation,
        "route_source": "direct_enhanced_answer",
        "quality_bypass": True
    })
    return state

def get_final_decision(state):
    """Edge function to extract the decision from state"""
    return state.get("decision", "useful")  # Default to useful if not set

# =============================================================================
# LANGGRAPH WORKFLOW CREATION WITH ENHANCED FALLBACK
# =============================================================================

print("🔧 LangGraph workflow is being created...")

# Create workflow
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("web_search", web_search)
workflow.add_node("web_search_fallback", web_search_fallback)
workflow.add_node("retrieve", retrieve) 
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
# Add separate nodes for hallucination and answer quality checking
workflow.add_node("check_hallucination", check_hallucination)
workflow.add_node("grade_answer_quality", grade_answer_quality)
workflow.add_node("decide_final_action", decide_final_action)
# Add direct web search node for quality-based routing
workflow.add_node("direct_web_search", direct_web_search)

# Connect graph edges
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "retrieve": "retrieve",
    },
)

# Web search and retrieval workflows
workflow.add_edge("web_search", "generate")
workflow.add_edge("web_search_fallback", "generate")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_after_grading,
    {
        "transform_query": "transform_query", 
        "generate": "generate",
        "retrieve": "retrieve",
        "web_search": "web_search"
    },
)

workflow.add_edge("transform_query", "retrieve")

# New workflow: generate -> hallucination check -> [decision] -> answer quality/web search -> final decision
workflow.add_edge("generate", "check_hallucination")

# Add conditional edge after hallucination check
workflow.add_conditional_edges(
    "check_hallucination",
    decide_after_hallucination_check,
    {
        "grade_answer_quality": "grade_answer_quality",
        "web_search": "web_search"
    }
)

# Replace simple edge with conditional edge for two-layer quality evaluation routing
workflow.add_conditional_edges(
    "grade_answer_quality",
    decide_after_quality_check,
    {
        "decide_final_action": "decide_final_action",
        "direct_web_search": "direct_web_search"
    }
)

# Direct web search provides final answer, so it goes directly to END
workflow.add_edge("direct_web_search", END)

workflow.add_conditional_edges(
    "decide_final_action",
    get_final_decision,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
        "web_search_fallback": "web_search_fallback",  # Web search fallback route
    },
)

# Compile workflow
graph = workflow.compile()

print("✅ Adaptive RAG workflow created successfully!")

# =============================================================================
# TEST FUNCTIONS
# =============================================================================

# Export graph
__all__ = ["graph"]

# LangGraph Studio ready - no terminal testing
print("\n✅ LangGraph workflow ready for Studio!")
print("🚀 Start with: langgraph dev --allow-blocking")