# Import sys for system path
import sys

# ============================================================================================================================
seed = 10
max_new_tokens = 1024
int8_threshold = 4.0

TEMP = 0.1
TOP_P = 0.1

    # TOP-K: {1, 2, 4}
retrieval_top_k = 4
chunk_size = 250
chunk_overlap = 0


log_name = str(max_new_tokens) + '_' + str(int8_threshold) + '_k' + str(retrieval_top_k) + '_chunk' + str(chunk_size) + '.txt'

asa_data_path = '<PATH_TO_THE_350_QUESTIONS_CSV_DATASET>'
result_log = '<PATH_FOR_FULL_OUTPUT_LOG_IN_TXT>' + log_name

# Setup LLM model
    # Model 0: Qwen/Qwen2.5-7B-Instruct
    # Model 1: meta-llama/Meta-Llama-3.1-8B-Instruct 
    # Model 2: Qwen/Qwen2.5-72B-Instruct 
model_name = "Qwen/Qwen2.5-72B-Instruct"
# Setup embedding model
  # RAG 0: dunzhang/stella_en_400M_v5
  # RAG 1: pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
  # RAG 2: abhinand/MedEmbed-large-v0.1
  # RAG 3: "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
embedding_model_name = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"

# Store the vector database path
embedding_store = '<PATH_TO_VECTOR_DATABASE_FOLDER>' + '_chunk'+ str(chunk_size) +'_overlap' + str(chunk_overlap)

general_log_file = open(result_log, 'w') # Open log file
# ============================================================================================================================
# Import required modules for LLM & RAG
import os
import time
import torch as pt  # ML biggest framework (Especially for gpu)
import numpy as np
import json
import pandas as pd # csv in/out handling
from glob import glob
from typing import List
from pprint import pprint

# Graph related modules
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import END, StateGraph, START

# Additional modules for grader and router
from langchain_core.prompts import PromptTemplate # Create an instruct template
from langchain_community.vectorstores import Chroma # For store vectorized documents and ez retrieve of reference text
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser # To parse output

# LLM related modules
from llama_index.llms.huggingface import HuggingFaceLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel, pipeline # In order to deploy llm
# ============================================================================================================================
# Setup inference device
#device = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")
print(pt.device)
#pt.set_default_device(device)
pt.backends.cudnn.deterministic = True

pt.cuda.manual_seed(seed)

# ============================================================================================================================
# This seperate retrieval explanation sentences is for `COMBINED` case
retrieved_docs = """1. Both drugs are short-acting and do not produce prolonged sedation and both drugs have little effect on REM sleep. Triazolam, like other benzodiazepines, produces anterograde amnesia.
2. Hypoxemia and intracranial hypertension are common problems after closed head trauma. Securing the airway and initiating mechanical ventilation may be required to maintain adequate oxygenation. Sedation, which decreases cerebral metabolic rate, is useful to reduce ICP but does obscure the neurological exam. Routine hyperventilation has been found not to be of benefit to outcome and is no longer recommended. The recommended lower limit for CPP is 60 mm Hg.
3. Meconium aspiration usually occurs in fullterm babies and is rare in those weighing less than 2 kg at birth. Regular chest physical therapy and postural drainage are recommended to clear residual meconium from the lungs. Long-term outcome is good, in terms of intellectual development and pulmonary function, unless asphyxia occurred in the perinatal period. Passage of meconium may occur in the presence or absence of fetal distress.
4. α1-antitrypsin deficiency leads to airway disease that is familial and is determined by serum assay. The assay measures the level of a protective enzyme produced by the liver that acts to prevent autodigestion of lung tissue by the proteolytic enzymes of phagocytic cells. Only 1-2% of COPD patients are found to have severe α1-antitrypsin deficiency as a contributing cause to their disease.
5. The morbidly obese patient is at a higher risk of having a macrosomic infant and a lower risk of preterm delivery. A higher initial failure rate of epidural placement has been reported and the need for placement of a second or third catheter is more common. Morbidly obese parturients are at higher risk for undiagnosed obstructive sleep apnea (OSA). Although the ASA guidelines for the perioperative management of patients with OSA was not intended specifically for pregnant patients, they do provide some guidance for the management of the obese parturient undergoing cesarean delivery.
6. The most likely finding on TEE based on the patient’s symptoms is severe mitral regurgitation (MR). MR in patients with HOCM occurs as a consequence of systolic anterior motion (SAM) of the anterior mitral valve leaflet, thus acutely narrowing the left ventricular outflow tract. This condition is exacerbated by states of increased contractility, in this case due to the infusion of epinephrine, and hypovolemia, for example due to sudden large volume blood loss. None of the other options listed would explain worsening hemodynamics in response to epinephrine infusion. TEE findings as described in option A can be expected in acute pulmonary embolus.
7. States of increased contractility worsen left ventricular outflow tract obstruction as described in the explanation to the previous question. Epinephrine should therefore be discontinued immediately. Milrinone and calcium would have the same effect and likely worsen this patient’s hemodynamics. The goal of therapy is restoration of preload with volume infusion and afterload support if indicated with pure α-adrenoceptor agonists such as phenylephrine. Nitroglycerin, due to the associated reduction in preload would likely worsen hemodynamics.
8. This previously healthy young patient has developed symptoms referable to both the chest and the head/ neck. Epiglottitis, pneumonia, or tracheomalacia could result in dyspnea, cough, and perhaps hoarseness, but they would be unlikely to cause facial swelling. Angioedema could account for this constellation of symptoms, but a three-day course would be unusual. This patient has symptoms of superior vena cava (SVC) syndrome, a clinical diagnosis based on symptomatology. Most cases of SVC syndrome are caused by malignancies such as lung cancer, lymphoma or metastases, but benign causes (strictures from intravascular devices, thyromegaly, and aortic aneurysm) also exist. SVC syndrome in a young man with a mediastinal mass is most commonly due to lymphoma or a mediastinal germ cell tumor. Patients with SVC syndrome commonly present with facial/ neck swelling, dyspnea, and cough. Hoarseness, headache, congestion, hemoptysis, dysphagia, pain and syncope may also be seen. Symptoms worsen with a head down position.
9. Although nonpharmacologic interventions (such as a quiet environment, soothing music, or a familiar family member at the bedside) may reduce the need for sedatives, many mechanically ventilated patients require sedation for safety. The choice of sedative agent should be tailored to the patient, keeping in mind the patient’s physiology and the side effect profiles of the sedative medications. Benzodiazepines can worsen confusion and delirium. As a result, their use should be limited to cases where side effects of other sedatives are unacceptable or where alcohol or benzodiazepine withdrawal may be contributing to confusion. Etomidate suppresses adrenocortical function. It is not appropriate for use as an infusion. Side effects of propofol include hypotension due to vasodilation, respiratory depression, and hyperlipidemia. Propofol infusion syndrome is a rare, but potentially fatal side effect of propofol administration. Dexmedetomidine is an α2-adrenoceptor agonist. Administration causes sedation and analgesia with little respiratory depression. Side effects of dexmedetomidine include hypotension and bradycardia. Dexmedetomidine is rarely used for long-term sedation (> 1 day) due to cost. In this patient with bradycardia and an acceptable blood pressure, propofol would be the most appropriate sedative. Pain should also be treated, as untreated pain is associated with an increased incidence of delirium and worse outcomes.
10. When advancing with a needle, the ligamentum flavum feels firm and crunchy in the midline. When the needle passes through the ligamentum flavum, the epidural space is encountered as a distinct loss of resistance.
"""

model_kwargs = {'trust_remote_code': True}

if not os.path.exists(embedding_store):
    print("Vectorstore not found. Creating new embeddings...")
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(embedding_tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap)    
    # Split docs into chunks
    texts_splits = text_splitter.split_text(retrieved_docs)
    print(texts_splits)
    # Create vector indexing for input docs
    vector_indexing = Chroma.from_texts(
        texts_splits,
        embedding=HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs),
        persist_directory=embedding_store
    )
    vector_indexing.persist()
else:
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(embedding_tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # Split docs into chunks
    texts_splits = text_splitter.split_text(retrieved_docs)
    print(texts_splits)
    print("Loading existing vectorized database...")
    vector_indexing = Chroma.from_texts(
        texts_splits,
        persist_directory=embedding_store,
        embedding=HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)    
    )

# ============================================================================================================================
# HuggingFace access token (if applicable)
access_token = "<HUGGINGE_FACE_ACCESS_TOKEN>"
# ============================================================================================================================
# Setup LLM model

# Set tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)

bnb_config = BitsAndBytesConfig(
    load_in_8bit = True,
    llm_int8_threshold = int8_threshold
)
llm_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 cache_dir='./model',
                                                 quantization_config=bnb_config,
                                                 token=access_token,
                                                 device_map = 'auto')

# ============================================================================================================================
# Retrieval
retriever = vector_indexing.as_retriever(search_kwargs={"k": retrieval_top_k})

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

# ============================================================================================================================
# Generate answer from retrieved document
def generate(user_prompt, gen_chain, retriever_em):
    print("---RETRIEVING....---")
    retrieved_docs = retriever_em.get_relevant_documents(user_prompt)
    general_log_file.write(f"\n\n\t{retrieved_docs}\n\n\t")
    print("---ANSWERING....---")
    generation = gen_chain.invoke({"question": (user_prompt + Answer_format), "context": retrieved_docs})
    print("---ANSWER GENERATED....---")
    return generation

# ============================================================================================================================
# Get ASA queries list
# Get ASA data - use 1 question first
asa_data = pd.read_csv(asa_data_path, sep = ';', engine = 'python')
print(asa_data.shape)


# ============================================================================================================================

    # Adjust runtime temp & top_p
llm_model_pipeline = pipeline(
    task="text-generation",
    model=llm_model,
    tokenizer=tokenizer,
    max_new_tokens = max_new_tokens,
    return_full_text=False,
    temperature = TEMP,
    top_p = TOP_P
)
hf_llm_model = HuggingFacePipeline(pipeline=llm_model_pipeline)

# Chain prompt with RAG to llm
rag_generate_chain = generate_prompt | hf_llm_model | StrOutputParser()

for i in range (0,asa_data.shape[0],1):
    # Prompt
    general_log_file.write(f"QUESTION {i}:\n")
    
    prompt = f"""{asa_data.iloc[i, 1]} \n
Choices: {asa_data.iloc[i, 2]}, {asa_data.iloc[i, 3]}, {asa_data.iloc[i, 4]}, {asa_data.iloc[i, 5]}, {asa_data.iloc[i, 6]} \n
Choose only one answer (A), (B), (C), (D) or (E) and give explaination in step by step.\n\n"""

    answer = generate(prompt, rag_generate_chain, retriever)

    pprint(answer)
    
    general_log_file.write(f"\t{answer}\n\n")
    general_log_file.write(f"==========================================================================\n")
    
    
general_log_file.close()