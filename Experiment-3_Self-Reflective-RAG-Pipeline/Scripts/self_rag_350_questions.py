# Import sys for system path
import sys

# Set `temperature` and `top_p`
  # NORMAL:     temp = {0.1; 1.0; 2.0}      | top_p = {0.1; 0.5; 1}
  # FINE:       temp = {0.05; 0.1; 0.5}     | top_p = {0.01; 0.05; 0.09}
  # VERY FINE:  temp = {0.005; 0.01; 0.05}  | top_p = {0.001; 0.005; 0.009}
TEMP = 0.1
TOP_P = 0.1

  # TOP_K = {4, 8, 12}
retrieval_top_k = 4

# Setup LLM model
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# Setup embedding model
  # RAG 0: dunzhang/stella_en_400M_v5
  # RAG 1: pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
  # RAG 2: abhinand/MedEmbed-large-v0.1
  # RAG 3: neuml/pubmedbert-base-embeddings
embedding_model = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"

# Setup path
documents_path = '<PATH_TO_THE_MILLER_PDF_TEXTBOOK>'
asa_data_path = '<PATH_TO_THE_350_QUESTIONS_CSV_DATASET>'
# Modify result path for exact file
asa_result_path = '<PATH_FOR_OUTPUT_ANSWERS_CSV>'
general_log_path = '<PATH_FOR_FULL_OUTPUT_LOG_IN_TXT>'

# ============================================================================================================================
# Import required modules for LLM & RAG
import os
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
# from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate # Create an instruct template
from langchain_community.vectorstores import Chroma # For store vectorized documents and ez retrieve of reference text
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser # To parse output

# LLM related modules
from llama_index.core import Settings # Setting the general context for querying - smilar to service context
from langchain.schema import Document
from llama_index.readers.file import PyMuPDFReader
from llama_index.llms.huggingface import HuggingFaceLLM # For interface between Settings and HuggingFace supportive model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline # In order to deploy llm

# ============================================================================================================================
# HuggingFace access token
access_token = "<HUGGINGE_FACE_ACCESS_TOKEN>"
os.environ["TAVILY_API_KEY"] = "<TAVILY_WEBSITE_API_KEY>"
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
# Create vector indexing for input docs
vector_indexing = Chroma.from_documents(
    docs_splits,
    embedding=HuggingFaceEmbeddings(model_name=embedding_model)
)

# ============================================================================================================================
# Set tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)

bnb_config = BitsAndBytesConfig(
    load_in_8bit = True,
    llm_int8_threshold = 4.0
)
llm_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="./model", quantization_config=bnb_config, token=access_token)
llm_model_pipeline = pipeline(
    task="text-generation",
    model=llm_model,
    tokenizer=tokenizer,
    return_full_text=False,
    temperature = TEMP,
    top_p = TOP_P
)
hf_llm_model = HuggingFacePipeline(pipeline=llm_model_pipeline)

# ============================================================================================================================
# Retrieval
retriever = vector_indexing.as_retriever(search_kwargs={"k": retrieval_top_k})

# ============================================================================================================================
# Retrieval Grader
grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance
    of a retrieved document to a user question. If the document contains keywords related to the user question,
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation. \n
    Do not include the documents and questions section in the answer.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
    partial_variables={"format_instructions": JsonOutputParser().get_format_instructions()},
)

retrieval_grader = grader_prompt | hf_llm_model | JsonOutputParser()

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
    input_variables=["question", "document"],
)

# Chain prompt with RAG to llm
rag_generate_chain = generate_prompt | hf_llm_model | StrOutputParser()

# ============================================================================================================================
# State Node

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# Retrieve ==> S1
def retrieve(state):
  """
  Retrieve documents
  Args:
    state (dict): The current graph state
  Returns:
    state (dict): New key added to state, documents, that contains retrieved documents
  """
  print("---RETRIEVE--- S1")
  question = state["question"] # Get question from state variable
  # Retrieval
  documents = retriever.invoke(question)
  return {"documents": documents, "question": question}

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# Grading retrieved documents ==> S2(0)
def grade_documents(state):
  """
  Determines whether the retrieved documents are relevant to the question
  If any document is not relevant, we will set a flag to run web search
  Args:
    state (dict): The current graph state
  Returns:
    state (dict): Filtered out irrelevant documents and updated web_search state
  """
  print("---CHECK DOCUMENT RELEVANCE TO QUESTION--- S2")
  question = state["question"]
  documents = state["documents"]
  search = "no"
  # Score each doc
  filtered_docs = []
  for d in documents:
    score = retrieval_grader.invoke({"question": question, "document": d.page_content})
    grade = score.get("score", "no")
    if grade.lower() == "yes": # This piece of docs are relevant
      print("++++++GRADE: DOCUMENT RELEVANT++++++")
      filtered_docs.append(d)
    else:                      # This piece of docs are not relevant
      print("++++++GRADE: DOCUMENT NOT RELEVANT++++++")
      continue
  # If no documents ==> Go for search
  if not filtered_docs:
    print("++++++NO DOCUMENT RELEVANT++++++")
    search = "yes"
  # Return decision result with relavent variable
  return {"documents": filtered_docs, "question": question, "search": search}

# ++++++++++++++++
# Generate decider ==> S2(1)
def decide_to_generate(state):
  """
  Determines whether to generate an answer, or re-generate a question.
  Args:
    state (dict): The current graph state
  Returns:
    str: Binary decision for next node to call
  """
  print("---ASSESS GRADED DOCUMENTS--- S3")  
  search = state["search"]
  if search.lower() == "yes":
  # We will re-generate a new query
    print("++++++DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, PERFORM WEBSEARCH++++++")
    return "web_search"
  else:
    print("++++++DECISION: GENERATE++++++")
  return "generate"

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# Doing web search in case no relavent documents were found ==> S4
web_search_tool = TavilySearchResults(k=3) # Use top-3 search

# V0 - Get metadata also
def web_search_v0(state):
  """
    Web search based on the re-phrased question.
  Args:
    state (dict): The current graph state
  Returns:
    state (dict): Updates documents key with appended web results
  """  
  web_results = []
  question = state["question"]
  documents = state.get("documents", [])
  web_results = web_search_tool.invoke({"query":question})
  documents.extend(
      [
          # Document(page_content=d["content"], metadata={"url": d["url"]})
          # for d in web_results
          Document(page_content=d.get("content", []), metadata={"url": d.get("url", [])})
          for d in web_results
      ]
  )
  return {"documents": documents, "question": question}

# V1 - Do not get metadata
def web_search_v1(state):
  """
    Web search based on the current question
  Args:
    state (dict): The current graph state
  Returns:
    state (dict): Updates documents key with appended web results
  """  
  print("---WEB SEARCH---")
  question = state["question"]
  documents = state["documents"]
  web_documents = []
  web_results = []

  # Web search
  pprint(question)
  for web_search_retry in (0,5,1):
    print(f"++++++WEB SEARCH RETRY {web_search_retry} ++++++")
    # Search
    web_docs = web_search_tool.invoke({"query": question})
    pprint(web_docs)
    
    for d in web_docs:
      if isinstance(d, dict):
        print("++++++THIS WEB SEARCH VALID ==> APPEND TO DOCUMENTS++++++")
        #web_results = "\n".join([d.get("content", [])])
        web_results = Document(page_content=d.get("content", []), metadata={"url": d.get("url", [])})
        web_documents.append(web_results)
      else:
        print("++++++THIS WEB SEARCH VALIE IS NOT VALID ==> DISCARD THIS ONE++++++")
        continue
        
    if web_documents == []:    
      print(f"++++++WEB SEARCH NOT VALID AT TRIAL {web_search_retry} ==> RETRY++++++")
      documents = []
    else:
      print(f"++++++WEB SEARCH VALID AT TRIAL {web_search_retry} ==> EXIT & RETURN RESULT++++++")
      documents.append(web_documents)
      break
    
    # Too much retries ==> quit
    if web_search_retry == 4 and web_results == []:
      print(f"++++++TOO MANY WEB SEARCH RETRIES, ABORT (retries: {web_search_retry}) ++++++")
      documents = []
      break
    

  return {"documents": documents, "question": question}

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# Check if the websearch result exist
def is_websearch_ok(state):
  """
    Web search based on the re-phrased question.
  Args:
    state (dict): The current graph state
  Returns:
    state (dict): Updates documents key with appended web results
  """
  documents = state.get("documents", [])
  question = state["question"]
  
  pprint("This is the document" + str(documents))
  
  if not documents:
    caution_answer = """For this question, you should add the following phrase at the end of the answer:\n
                        'No relavent document was found, therefore the answer is based solely on the model trained parameters!!!'"""
    question += caution_answer
    documents = []
    print("++++++WEB SEARCH RETURN [] RESULT++++++")
  else:
    print("++++++WEB SEARCH RETURN VALID RESULT++++++")
  
  return {"documents": documents, "question": question}

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# Generate answer from retrieved document
def generate(state):
  """
  Generate answer using RAG on retrieved documents
  Args:
    state (dict): The current graph state
  Returns:
    state (dict): New key added to state, generation, that contains LLM generation
  """
  print("---GENERATE--- S4")
  question = state["question"]
  documents = state["documents"]
  # RAG generation
  generation = rag_generate_chain.invoke({"question": (question + Answer_format), "context": documents})
  return {"documents": documents, "question": question, "generation": generation}

# ============================================================================================================================
class GraphState(TypedDict):
    """
    Represents the state of chain graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    search: str
    documents: List[str]
    
    
workflow = StateGraph(GraphState)

# Define 4 nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("web_search", web_search_v1)
workflow.add_node("is_websearch_ok", is_websearch_ok)

# Build graph ==> pipeline logic
workflow.add_edge(START, "retrieve") # Start with retrieve
workflow.add_edge("retrieve", "grade_documents") # Grade the retrieved documetns
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate": "generate",
    }
)
workflow.add_edge("web_search", "is_websearch_ok")
workflow.add_edge("is_websearch_ok", "generate")
workflow.add_edge("generate", END)

# Compile graph
self_rag_websearch_graph = workflow.compile()

# Draw graph
display(Image(self_rag_websearch_graph.get_graph(xray=True).draw_mermaid_png()))

# ============================================================================================================================
# Get ASA queries list
# Get ASA data - use 1 question first
asa_data = pd.read_csv(asa_data_path, sep = ';', engine = 'python')
# Append answer data
asa_data['Gen_Answer'] = None
asa_data['Gen_Explain'] = None

# ============================================================================================================================
# Test graph

print("RUN ASA QUERIES")
general_log_file = open(general_log_path, 'a')

for i in range (20,asa_data.shape[0],1):    
  # Compile ASA query from dataframe
  iter_prompt = f"""{asa_data.iloc[i, 1]} \n
  Choices: {asa_data.iloc[i, 2]}, {asa_data.iloc[i, 3]}, {asa_data.iloc[i, 4]} \n
  Choose only one answer (A), (B), (C), (D) or (E) and give explaination in step by step.\n\n"""
    
  inputs = {"question": iter_prompt}

  for output in self_rag_websearch_graph.stream(inputs):
    # Node monitoring
    for key, value in output.items():
      log_node = f"Node '{key}':"
      pprint(log_node)
      general_log_file.write(log_node)  
    
  # Log and write answer
  iter_answer = value["generation"]
  # To CSV file
  asa_data.iloc[i,7] = iter_answer
  asa_data.to_csv(asa_result_path, sep = ';', index = False)
  # To Log file
  pprint(iter_answer)
  print("Question " + str(i) + " done!")
  print("\n-------------------------------------------\n\n")
  general_log_file.write(iter_answer + "\nQuestion " + str(i) + " done!" + "\n\n-------------------------------------------\n\n")
    
general_log_file.close()
print("DONEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")