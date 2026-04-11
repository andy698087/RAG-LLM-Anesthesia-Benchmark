# Import sys for system path
import sys

TEMP = 0.1
TOP_P = 0.1

# Run index
run_index = 1

# Setup path
documents_path = '<PROJECT_ROOT>/textbook_retrieval_as_pdf/'
asa_data_path = '<PROJECT_ROOT>/questions/inout_data_350ques.csv'
# Modify result path for exact file
asa_result_path = '<PROJECT_ROOT>/complement_out/350-set/rag_0_llama3_0_k_k12_result_' + str(run_index) + f'_{TEMP}_{TOP_P}.csv'
general_log_path = '<PROJECT_ROOT>/complement_out/350-set/logs/rag_0_llama3_0_k_k12_result_' + str(run_index) + f'_{TEMP}_{TOP_P}.txt'

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
# HuggingFace access token
access_token = "<HF_ACCESS_TOKEN>"

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
    embedding=HuggingFaceEmbeddings(model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", model_kwargs=model_kwargs)
    #embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/average_word_embeddings_komninos")
    # embedding=HuggingFaceEmbeddings(model_name="dunzhang/stella_en_400M_v5", model_kwargs=model_kwargs)
)

# ============================================================================================================================
# Setup LLM model
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# Set tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)

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
Answer_format = """
Select ONE BEST option to answer or complete the statement and the desired answer should be in formulated in JSON format.
The JSON answer should contain 2 keys, key "Answer" correspond to answer choice (A, B, C, D or E) while key "Explain" contain the short step-by-step explaination.
The two keys ("Answer" and "Explain") and the answer letter A, B, C, D, E (and A, B, C, D, E only, do not put the content of the selectedd choice) after the key "Answer" should be wrapped in double quote "" to form a more standard JSON format.
The explain sentence after the key "Explain" should also be wrapped in double quotes "".
The sample format for JSON answer is as follow: 
{
  Answer: <Selected choice>,
  Explain: <step-by-step explainatioin>
}"""

generate_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an anesthesiologist having an exam. Please answer this question according to the context information.<|eot_id|><|start_header_id|>user<|end_header_id|>
Context: {context}
Question: {question}
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
    # Compile ASA query from dataframe
    iter_prompt = f"""{asa_data.iloc[i, 1]} \n
Choices: {asa_data.iloc[i, 2]}, {asa_data.iloc[i, 3]}, {asa_data.iloc[i, 4]}, {asa_data.iloc[i, 5]}, {asa_data.iloc[i, 6]} \n
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