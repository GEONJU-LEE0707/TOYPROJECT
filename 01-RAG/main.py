import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import yaml
from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.prompts import loading
from langchain_core.prompts.base import BasePromptTemplate
import glob


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


if "messages" not in st.session_state:
    load_dotenv()
    # 세션 메시지 초기화
    st.session_state.messages = []

# system_prompt = "당신은 친절하게 대답하는 어시스턴트 입니다."

## 사이드바 생성
with st.sidebar:
    # 대화 초기화
    clear_button = st.button("🗑️ 대화 초기화")
    if clear_button:
        st.session_state.messages = []
    # system prompt 설정
    promt_filepath = st.selectbox(
        "원하는 시스템 프롬프트를 선택해주세요",
        glob.glob("prompts/*.yaml"),
        index=0,
    )
    task = st.selectbox(
        "원하는 작업을 선택해주세요",
        ["요약", "번역", "SNS"],
        index=0,
    )

    st.write("당신이 선택한 시스템 프롬프트:", promt_filepath)


## 상단 타이틀
st.title("💬 Chatbot")


# 이전 대화 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지 추가
def add_message(role, content):
    st.session_state.messages.append(ChatMessage(role=role, content=content))


# 체인 생성
def create_chain(promt_filepath):
    # prompt | llm | output_parser

    prompt = load_prompt(promt_filepath, encoding="utf-8")

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    return chain


print_messages()

if user_input := st.chat_input("질문을 입력하세요"):
    # 사용자의 입력 출력
    st.chat_message("user").write(user_input)

    # 체인 생성
    chain = create_chain(promt_filepath=promt_filepath)
    response = chain.stream({"question": user_input, "task": task})
    with st.chat_message("assistant"):
        # 빈 공간(컨테이너)를 만들어서, 여기에 토큰을 스트리밍 출력
        container = st.empty()
        ai_answer = ""
        for token in response:
            ai_answer += token
            container.write(ai_answer)

    # ai_answer = chain.invoke({"question": user_input})

    # 답변 출력
    # st.chat_message("assistant").write(ai_answer)

    # 대화 저장
    add_message("user", user_input)
    add_message("assistant", ai_answer)
