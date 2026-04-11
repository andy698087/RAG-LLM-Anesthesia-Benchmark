# Import sys for system path
import sys

# ============================================================================================================================
seed = 10
retrieval_top_k = 1

percentile = 95.0

asa_data_path = '<PATH_TO_QUESTION_6_IN_10_QUESTION_CSV>'

embedding_model_name = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"

documents_path = '<PATH_TO_MILLER_TEXTBOOK>'
embedding_store = '<PATH_TO_VECTOR_DATABASE_FOLDER>' + '_chunk' + str(percentile)

# ============================================================================================================================
# Import required modules for LLM & RAG

import os
import time
import torch as pt  # ML biggest framework (Especially for gpu)
import numpy as np
import json
import pandas as pd # csv in/out handling
from glob import glob
from pprint import pprint

# LLM related modules
from langchain.schema import Document

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, AutoModelForCausalLM # In order to deploy llm

# ============================================================================================================================
# Setup inference device
print(pt.device)
pt.backends.cudnn.deterministic = True
pt.cuda.manual_seed(seed)

# ============================================================================================================================

# Documents loader
pdf_loader = PyPDFLoader(documents_path)
documents = pdf_loader.load()

# Setup embed model: Can use either: FastEmbedEmbeddings or HuggingFaceEmbeddings | Some models are not supported in one of those APIs
embed_model_kwargs = {'trust_remote_code': True}
embed_model = HuggingFaceEmbeddings(model_name=embedding_model_name,
                                    model_kwargs=embed_model_kwargs)

if not os.path.exists(embedding_store):
    print("Vectorstore not found. Creating new embeddings...")
    
    # Documents loader
    pdf_loader = PyPDFLoader(documents_path)
    documents = pdf_loader.load()
    # Create semantic chunks
    semantic_chunker = SemanticChunker(embed_model,
                                       breakpoint_threshold_type="percentile",
                                       breakpoint_threshold_amount=percentile)
    semantic_chunks = semantic_chunker.create_documents([d.page_content for d in documents])
    
    # Store embedding vectors
    vector_indexing = Chroma.from_documents(
        semantic_chunks,
        embedding = embed_model,
        persist_directory=embedding_store
    )    
    vector_indexing.persist()
else:
    # Load save embedding vectors
    print("Loading existing vectorized database...")
    vector_indexing = Chroma(
        persist_directory = embedding_store,
        embedding_function = embed_model
    )

# ============================================================================================================================
# Retrieval
retriever = vector_indexing.as_retriever(search_kwargs={"k": retrieval_top_k})

# ============================================================================================================================
# Get ASA queries list
# Get ASA data - use 1 question first
asa_data = pd.read_csv(asa_data_path, sep = ';', engine = 'python')

# ============================================================================================================================
for i in range (0,asa_data.shape[0],1):
    # Prompt
    print(f"QUESTION {i}:\n")
    
    prompt = f"""{asa_data.iloc[i, 1]} \n
Choices: {asa_data.iloc[i, 2]}, {asa_data.iloc[i, 3]}, {asa_data.iloc[i, 4]}, {asa_data.iloc[i, 5]}, {asa_data.iloc[i, 6]} \n
Choose only one answer (A), (B), (C), (D) or (E) and give explaination in step by step.\n\n"""

    retrieval_docs = retriever.get_relevant_documents(prompt)

    pprint(retrieval_docs)