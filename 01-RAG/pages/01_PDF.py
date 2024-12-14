import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages.chat import ChatMessage
import yaml
from langchain_core.prompts import loading
from langchain_core.prompts.base import BasePromptTemplate
from langchain_teddynote import logging
import os

# API 키 로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("PDF RAG")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

## 상단 타이틀
st.title("PDF 기반 RAG")


def load_prompt(file_path, encoding="utf8") -> BasePromptTemplate:
    """
    파일 경로를 기반으로 프롬프트 설정을 로드합니다.

    이 함수는 주어진 파일 경로에서 YAML 형식의 프롬프트 설정을 읽어들여,
    해당 설정에 따라 프롬프트를 로드하는 기능을 수행합니다.

    Parameters:
    file_path (str): 프롬프트 설정 파일의 경로입니다.

    Returns:
    object: 로드된 프롬프트 객체를 반환합니다.
    """
    with open(file_path, "r", encoding=encoding) as f:
        config = yaml.safe_load(f)

    return loading.load_prompt_from_config(config)


if "chain" not in st.session_state:
    st.session_state.chain = None

if "messages" not in st.session_state:
    load_dotenv()
    # 세션 메시지 초기화
    st.session_state.messages = []


## 사이드바 생성
with st.sidebar:

    selected_model = st.selectbox(
        "모델을 선택해주세요", ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"], index=0
    )
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
    st.write(uploaded_file)
    if uploaded_file:
        st.write(uploaded_file.name)

    clear_button = st.button("🗑️ 대화 초기화")
    if clear_button:
        st.session_state.messages = []


# 이전 대화 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지 추가
def add_message(role, content):
    st.session_state.messages.append(ChatMessage(role=role, content=content))


# 파일이 업로드되면 캐시 디렉토리에 저장
@st.cache_resource(show_spinner="업로드 중...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    ##################################
    ### RAG STEP 1. Retriever 생성 ###
    ##################################

    # 단계 1: 문서 로드(Load Documents)
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    return retriever


# 체인 생성
def create_chain(retriever, model_name):

    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = load_prompt("prompts/pdf-rag.yaml")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# 파일이 업로드되면, retriever 생성 (보통 오래걸림)
if uploaded_file:
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever=retriever, model_name=selected_model)
    st.session_state.chain = chain

# 경고 메시지를 띄우기 위한 코드
warning_message = st.empty()

if user_input := st.chat_input("질문을 입력하세요"):

    # 체인 생성
    chain = st.session_state.chain
    if chain is not None:
        # 사용자의 입력 출력
        st.chat_message("user").write(user_input)
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)를 만들어서, 여기에 토큰을 스트리밍 출력
            container = st.empty()
            ai_answer = ""
            for token in response:
                ai_answer += token
                container.write(ai_answer)

        # 대화 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        warning_message.error("파일을 업로드 해주세요")
