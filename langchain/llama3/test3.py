from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.retrievers  import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA
import time
import chainlit as cl

start = time.time()

pdf_filepath = 'test_samsung.pdf'
loader = PyPDFLoader(pdf_filepath)
pages = loader.load()
print("=====loader=====")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap  = 50,
    length_function = len,
)
print("=====text_splitter=====")

# texts = text_splitter.split_text(pages)
chunks = text_splitter.split_documents(pages)
print("=====chunks=====")
embeddings = OllamaEmbeddings(model="ggml-model-Q4_K_M:latest")
print("=====embeddings=====")
# vectorstore = FAISS.from_documents(chunks,embedding = embeddings,distance_strategy = DistanceStrategy.COSINE)
# print("=====vectorstore=====")

# 로컬에 DB 저장
# MY_FAISS_INDEX = "./MY_FAISS_INDEX_0609"
# vectorstore.save_local(MY_FAISS_INDEX)

# vectorstore = FAISS.load_local(MY_FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)

corpus=  [i.page_content for i in chunks]

bm25_retriever = BM25Retriever.from_texts(corpus)
bm25_retriever.k = 6

faiss_vector = FAISS.from_texts(corpus, embeddings)
faiss_retriever = faiss_vector.as_retriever(search_kwargs={'k':5})

ensemble_retriever = EnsembleRetriever(
                    retrievers = [bm25_retriever,faiss_retriever]
                    , weight = {0.2,0.8})
print("=====ensemble_retriever=====")
# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}) # 유사도 높은 5문장 추출



query = '요약 손익 계산서 테이블에서 22년도 3분기의 총 매출액이 얼마야?'
# query = '핸드셋 관련 매출액과 가이던스 그래프를 요약해서 알려줘 

llm = ChatOllama(model="Llama-3-Open-Ko-8B-Q8_0:latest")

qa = RetrievalQA.from_chain_type(
    llm = llm
    , chain_type='refine'
    , retriever = ensemble_retriever)

res = qa(query)
print(res['result'])
docs = ensemble_retriever.get_relevant_documents(query)
# print("=====retriever=====")

##################################################################### 멀티쿼리
# from langchain.retrievers.multi_query import MultiQueryRetriever
# llm = ChatOllama(model="Llama-3-Open-Ko-8B-Q8_0:latest")
# retriever_from_llm = MultiQueryRetriever.from_llm(
#     retriever=vectorstore.as_retriever(), llm=llm
# )
# import logging

# logging.basicConfig()
# logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

# unique_docs = retriever_from_llm.get_relevant_documents(query=query)
###########################################################################
#############################################################Contextual compression
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import LLMChainExtractor

# llm = ChatOllama(model="Llama-3-Open-Ko-8B-Q8_0:latest")
# base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}) # 유사도 높은 5문장 추출
# compressor = LLMChainExtractor.from_llm(llm)
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor, base_retriever=base_retriever
# )

# compressed_docs = compression_retriever.get_relevant_documents(query)
#####################################################################################

# Prompt
# template = '''Answer the question based only on the following context:
# {context}

# Question: {question}
# '''
template = '''밑의 질문에 맞추어서 답변을 생성해줘. 대신 넌 금융 전문가인 것을 기억해.
{context}

질문: {question}
'''

prompt = ChatPromptTemplate.from_template(template)
# prompt = hub.pull("rlm/rag-prompt")

# Model

# llm = ChatOllama(model="llama3:latest")

# 한국어 특화 모델

print("=====llm=====")

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

# Chain
chain = prompt | llm | StrOutputParser()

# Run
response = chain.invoke({'context': (format_docs(docs)), 'question':query})
print("기존꺼",response)

print("총 걸린 시간 : ",time.time() - start)
