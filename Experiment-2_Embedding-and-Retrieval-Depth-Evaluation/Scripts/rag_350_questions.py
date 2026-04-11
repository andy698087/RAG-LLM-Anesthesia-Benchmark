# Import sys for system path
import sys

# Set `temperature` and `top_p`
  # NORMAL:     temp = {0.1; 1.0; 2.0}      | top_p = {0.1; 0.5; 1}
  # FINE:       temp = {0.05; 0.1; 0.5}     | top_p = {0.01; 0.05; 0.09}
  # VERY FINE:  temp = {0.005; 0.01; 0.05}  | top_p = {0.001; 0.005; 0.009}
TEMP = 0.1
TOP_P = 0.1

# Setup LLM model
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# Setup embedding model
  # RAG 0: dunzhang/stella_en_400M_v5
  # RAG 1: pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
  # RAG 2: abhinand/MedEmbed-large-v0.1
  # RAG 3: neuml/pubmedbert-base-embeddings
embedding_model = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"

# Run index
run_index = 1

# Setup path
documents_path = '<PATH_TO_THE_MILLER_PDF_TEXTBOOK>'
asa_data_path = '<PATH_TO_THE_350_QUESTIONS_CSV_DATASET>'
# Modify result path for exact file
asa_result_path = '<PATH_FOR_OUTPUT_ANSWERS_CSV>'
general_log_path = '<PATH_FOR_FULL_OUTPUT_LOG_IN_TXT>'

# HuggingFace access token
access_token = "<HUGGING_FACE_ACCESS_TOKEN>"

# ============================================================================================================================
# Import required modules for LLM & RAG
import os
import time
import torch as pt  # ML biggest framework (Especially for gpu)
import pandas as pd # csv in/out handling
from glob import glob
from typing import List
from pprint import pprint

# Graph related modules
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import END, StateGraph, START

# Web Search modules
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import TavilySearchResults

# Additional modules for grader and router
from langchain_core.prompts import PromptTemplate # Create an instruct template
from langchain_community.vectorstores import Chroma # For store vectorized documents and ez retrieve of reference text
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser # To parse output

# LLM related modules
from langchain.schema import Document
from llama_index.readers.file import PyMuPDFReader
from llama_index.llms.huggingface import HuggingFaceLLM # For interface between Settings and HuggingFace supportive model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline # In order to deploy llm

# ============================================================================================================================
# Setup inference device
device = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")
print(device)
pt.set_default_device(device)
print(pt.get_default_device())

# ============================================================================================================================
# Documents loader
pdf_loader = PyMuPDFReader()

# Documents loading function
def load_documents(directory):
  documents_list = []
  for item_path in glob(directory + "*.pdf"):
    # The loaded document from PyMuReader have to be converted into "Document" object according to langchain
    for page in pdf_loader.load(file_path=item_path, metadata=True):
      documents_list.append(Document(page_content=page.text, metadata=page.metadata))
    print(documents_list)
  return documents_list

# Setup Documents text splitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)

# Get docs
docs = load_documents(documents_path)
# Split docs into chunks
docs_splits = text_splitter.split_documents(docs)
# Allow external embedding model to run
model_kwargs = {'trust_remote_code': True}
# Create vector indexing for input docs
vector_indexing = Chroma.from_documents(
    docs_splits,
    embedding=HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs=model_kwargs)
)

# ============================================================================================================================


# Set tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)

# 8-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit = True,
    llm_int8_threshold = 4.0
)
llm_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 cache_dir='./model',
                                                 quantization_config=bnb_config,
                                                 token=access_token,
                                                 device_map = 'cuda')
llm_model_pipeline = pipeline(
    task="text-generation",
    model=llm_model,
    tokenizer=tokenizer,
    max_new_tokens = 1024,
    return_full_text=False,
    temperature = TEMP,
    top_p = TOP_P
)
hf_llm_model = HuggingFacePipeline(pipeline=llm_model_pipeline)

# ============================================================================================================================
# Retrieval
retriever = vector_indexing.as_retriever(search_kwargs={"k": 12})

# ============================================================================================================================
# Generate
Answer_format = """\n\nThe desired answer should be in formulated in JSON format.
The JSON answer should contain 2 keys, key 'answer' correspond to answer choice (A, B, C, D, or E) while key 'Explain' contain the short step-by-step explaination. \n
The sample format for JSON answer is as follow: \n
{\n
  'Answer': <Selected choice>, \n
  'Explain'": <step-by-step explainatioin>\n
}"""

generate_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question}
    Context: {context}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context"],
)

# Chain prompt with RAG to llm
rag_generate_chain = generate_prompt | hf_llm_model | StrOutputParser()

# ============================================================================================================================
# Generate answer from retrieved document
def generate(user_prompt):
    print("---RETRIEVING....---")
    retrieved_docs = retriever.get_relevant_documents(user_prompt)
    print("---ANSWERING....---")
    generation = rag_generate_chain.invoke({"question": (user_prompt + Answer_format), "context": retrieved_docs})
    print("---ANSWER GENERATED....---")
    return generation

# ============================================================================================================================
# Get ASA queries list
# Get ASA data - use 1 question first
asa_data = pd.read_csv(asa_data_path, sep = ';', engine = 'python')
# Append answer data
asa_data['Gen_Answer'] = None
asa_data['Gen_Explain'] = None
asa_data['Run_Time'] = None

# ============================================================================================================================

print("============================================================================================================")
print("============================================================================================================")
print("============================================================================================================")
print("============================================================================================================")
print("RUN ASA QUERIES")
general_log_file = open(general_log_path, 'a')
general_log_file.write('RUN INDEX: ' + str(run_index))
general_log_file.write("--- LET START ---")

for i in range (0,asa_data.shape[0],1):    
    iter_prompt = f"""{asa_data.iloc[i, 1]} \n
    Choices: {asa_data.iloc[i, 2]}, {asa_data.iloc[i, 3]}, {asa_data.iloc[i, 4]} \n
    Choose only one answer (A), (B), (C), (D) or (E) and give explaination in step by step.\n\n"""        
    # Get answer
    start_time = time.time_ns() // 1_000_000
    iter_answer = generate(iter_prompt)
    stop_time = time.time_ns() // 1_000_000
    # Calculate generated time
    time_diff = stop_time - start_time    
    # To CSV file
    asa_data.iloc[i,7] = iter_answer
    asa_data.iloc[i,8] = f"{time_diff}"
    asa_data.to_csv(asa_result_path, sep = ';', index = False)
    # To Log file
    pprint(iter_answer)
    print("Question " + str(i) + f" done! (in {time_diff})")
    print("\n-------------------------------------------\n\n")
    general_log_file.write(iter_answer + "\nQuestion " + str(i) + f" done! in ({time_diff})" + "\n\n-------------------------------------------\n\n")
    
general_log_file.close()
print("DONEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")