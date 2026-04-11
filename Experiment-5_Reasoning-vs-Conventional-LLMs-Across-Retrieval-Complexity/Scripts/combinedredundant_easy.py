# Import sys for system path
import sys
import importlib.util
# ============================================================================================================================
    # Initialize hyper settings
diff_level = "easy"
test_type = "combinedredundant"
max_new_tokens = 1024
int8_threshold = 4.0

    # LLM model
        # Model 0: "meta-llama/Meta-Llama-3-8B-Instruct"
        # Model 1: "meta-llama/Meta-Llama-3.1-8B-Instruct"
        # Model 2: "meta-llama/Llama-3.2-3B-Instruct"
        # Model 3: "meta-llama/Llama-3.3-70B-Instruct"
        # Model 4: "Qwen/Qwen2.5-72B-Instruct"
        # Model 5: "Qwen/qwen_3.0_8B"
        # Model 6: "Qwen/qwen_3.0_32B"
model_name = "Qwen/Qwen2.5-72B-Instruct"

    # RAG embedding Hyper-parameters
rag_setting = 0
percentile = 95.0
retrieval_top_k = 1
chunk_metric = "percentile"
embedding_model_name = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"

# ============================================================================================================================
# Import required modules for LLM & RAG
import os
import time
import json
import torch as pt  # ML biggest framework (Especially for gpu)
import numpy as np
import pandas as pd # csv in/out handling
from glob import glob
from typing import List
from pprint import pprint


# Additional modules for grader and router
from langchain_core.prompts import PromptTemplate # Create an instruct template
from langchain_community.vectorstores import Chroma # For store vectorized documents and ez retrieve of reference text
from langchain_core.output_parsers import StrOutputParser # To parse output

# LLM related modules
from langchain.schema import Document
from llama_index.llms.huggingface import HuggingFaceLLM # For interface between Settings and HuggingFace supportive model
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel, pipeline # In order to deploy llm

# ============================================================================================================================
# Load explanation file
exp_big_str = f"<PROJECT_ROOT>/combined_redundant_{diff_level}.py"
spec = importlib.util.spec_from_file_location("retrieval_docs", exp_big_str)
retrieval_docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrieval_docs)

# ============================================================================================================================
# Setup inference device
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print(device)
pt.set_default_device(device)
pt.backends.cudnn.deterministic = True

# ============================================================================================================================
# HuggingFace access token (if applicable)
access_token = "<HUGGINGE_FACE_ACCESS_TOKEN>"

# ============================================================================================================================
# Store path of vector database
embedding_store = f'<PROJECT_ROOT>/semantic_rag{rag_setting}_{chunk_metric}{percentile}_level_{diff_level}'

    # Embedding model
embed_model_kwargs = {'trust_remote_code': True}
embed_model = HuggingFaceEmbeddings(model_name=embedding_model_name,
                                    model_kwargs=embed_model_kwargs)

if not os.path.exists(embedding_store):
    print("Vectorstore not found. Creating new embeddings...")
    
    semantic_chunker = SemanticChunker(embed_model,
                            breakpoint_threshold_type=chunk_metric,
                            breakpoint_threshold_amount=percentile)
    texts_splits = semantic_chunker.split_text(retrieval_docs.retrieved_docs)
    
    # Create vector indexing for input docs
    vector_indexing = Chroma.from_texts(
        texts_splits,
        embedding=embed_model,
        persist_directory=embedding_store
    )
    vector_indexing.persist()
else:    
    print("Loading existing vectorized database...")
    vector_indexing = Chroma.from_texts(
        persist_directory=embedding_store,
        embedding=embed_model
    )

# Retriever
retriever = vector_indexing.as_retriever(search_kwargs={"k": retrieval_top_k})

# ============================================================================================================================

# Setup LLM model

# Set tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)

bnb_config = BitsAndBytesConfig(
    load_in_8bit = True,
    llm_int8_threshold = int8_threshold
)
llm_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 cache_dir='./model/',
                                                 quantization_config=bnb_config,
                                                 token=access_token,
                                                 device_map = 'auto')

# ============================================================================================================================
# Generate
Answer_format = """Select ONE BEST option to answer or complete the statement and respond only with valid JSON. Do not write an introduction or summary. The JSON answer contains 2 keys, key "Answer" correspond to answer option (A, B, C, D or E) while key "Explain" contains a short step-by-step explaination. If the option E is "THIS IS NOT AN ANSWER", please do not select E option.
The sample format is as follow:
{
    Answer: <Selected choice>,
    Explain: <step-by-step explaination>
}"""

generate_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an anesthesiologist having an exam. Please answer this question according to the context information.
<|eot_id|><|start_header_id|>user<|end_header_id|>Context: {context}
Question: {question}
Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context"],
)

# ============================================================================================================================
    # Adjust runtime temp & top_p
llm_model_pipeline = pipeline(
    task="text-generation",
    model=llm_model,
    tokenizer=tokenizer,
    max_new_tokens = max_new_tokens,
    return_full_text=False,
    temperature = 0.1,
    top_p = 0.1
)
hf_llm_model = HuggingFacePipeline(pipeline=llm_model_pipeline)

# Chain prompt with RAG to llm
rag_generate_chain = generate_prompt | hf_llm_model | StrOutputParser()

# ============================================================================================================================
# Generate answer with corresponded explanation
def generate(user_prompt, gen_chain, retriever):        
    print("---ANSWERING....---")
    retrieved_docs = retriever.get_relevant_documents(user_prompt)
    model_input = user_prompt + Answer_format
    print(f"INPUT PROMPT TO MODEL: {model_input}\n")
    generation = gen_chain.invoke({"question": model_input, "context": retrieved_docs})
    print("---ANSWER GENERATED....---")
    return generation

# ============================================================================================================================
# Test on all 3 sets
for ques_num in range(1,4,1):
    # Setup paths for read & write
    log_name = f"{test_type}_{max_new_tokens}_{int8_threshold}_log_ques{ques_num}_level_{diff_level}"
    result_log = f'<PROJECT_ROOT>/{test_type}/{log_name}.txt'
    asa_data_path = f'<PROJECT_ROOT>/ques_100_set_{ques_num}_w_exp.csv'
    answer_csv_path = f'<PROJECT_ROOT>/data_result_formal/{test_type}/{log_name}.csv'
    # Open the file for result log!
    general_log_file = open(result_log, 'w') # Open log file

    # Get ASA queries list
        # Read file
    asa_data = pd.read_csv(asa_data_path, sep = ';', engine = 'python')
            # Create new column
    asa_data['Gen_Answer'] = None
    asa_data['Gen_Explain'] = None
    asa_data['Run_Time'] = None
        # Shape of input file
    print(asa_data.shape)

    # Print the set NUMBER
    print(f"\n\nQUESTION SET {ques_num} |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| QUESTION SET {ques_num}\n\n")


    for i in range (0,asa_data.shape[0],1):
        # Prompt
        general_log_file.write(f"QUESTION {i}:\n")
    
        # Compile ASA query from dataframe
        prompt = f"""{asa_data.iloc[i, 1]}
Choices: {asa_data.iloc[i, 2]}, {asa_data.iloc[i, 3]}, {asa_data.iloc[i, 4]}, {asa_data.iloc[i, 5]}, {asa_data.iloc[i, 6]}
Choose only one answer (A), (B), (C), (D) or (E) and give explaination in step by step.\n"""

        # Generate answer with timestampt
        start_time = time.time_ns() // 1_000_000
        iter_answer = generate(prompt, rag_generate_chain, retriever)
        stop_time = time.time_ns() // 1_000_000
        time_diff = stop_time - start_time

        # Try to extract the seclected answer
        data = ''
        try:
            data = json.loads(iter_answer)
        except:
            data = None
    
        if data != None:
            selected_answer = data.get('Answer', '')
            asa_data.iloc[i,9] = selected_answer
            
        # To CSV file
        asa_data.iloc[i,10] = iter_answer
        asa_data.iloc[i,11] = f"{time_diff}"
        asa_data.to_csv(answer_csv_path,sep = ";",index=False)

        # Output answer to slurm out file system
        pprint(iter_answer)    
    
        # Output answer to log file
        general_log_file.write(f"\t{iter_answer}\n\n")
        general_log_file.write(f"==========================================================================\n")
        
    general_log_file.close()
