import streamlit as st
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

from langchain_teddynote import logging
logging.langsmith("English Chat AI")


from langchain_openai import ChatOpenAI
# from langsmith import Client

# Streamlit 애플리케이션 설정
st.set_page_config(page_title="AI English Chat", layout="wide")



# 사용자 설정 변수 초기화
if "user_settings" not in st.session_state:
    st.session_state["user_settings"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "waiting_for_response" not in st.session_state:
    st.session_state["waiting_for_response"] = False  # 응답 대기 상태


# 초기 화면: 사용자 설정
if st.session_state["user_settings"] is None:
    st.title("Welcome to AI English Chat!")
    st.write("Practice English tailored to your level, age, and preferences. 🧑‍🏫")
    
    # 사용자 설정 입력
    col1, col2, col3 = st.columns(3)
    with col1:
        level = st.selectbox("English Level", ["Beginner", "Intermediate", "Advanced"])
    with col2:
        age = st.selectbox("Age Group", ["10s", "20s", "30s", "40+"])
    with col3:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    if st.button("Start Chatting"):
        st.session_state["user_settings"] = {"level": level, "age": age, "gender": gender}
        st.success("Settings saved! Start chatting below.")
        # st.experimental_rerun()

# 채팅 화면
else:
    st.title("AI English Chat")
    st.write(f"**Level:** {st.session_state['user_settings']['level']} | "
             f"**Age:** {st.session_state['user_settings']['age']} | "
             f"**Gender:** {st.session_state['user_settings']['gender']}")
    
    # 이전 채팅 기록 표시
    if st.session_state["chat_history"]:
        for message in st.session_state["chat_history"]:
            if message["role"] == "user":
                st.write(f"**You:** {message['content']}")
            else:
                st.write(f"**AI:** {message['content']}")
    # 응답 대기 중 표시
    if st.session_state["waiting_for_response"]:
        st.info("Generating a response... Please wait.")
        
    # 사용자 입력
    user_input = st.text_input("Your message", disabled=st.session_state["waiting_for_response"])
    if st.button("Send", disabled=st.session_state["waiting_for_response"]):
        if user_input.strip():
            # 사용자 입력 저장
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            st.session_state["waiting_for_response"] = True  # 응답 대기 상태 활성화
            
            # OpenAI API 호출
            try:
                llm = ChatOpenAI(
                temperature=0.1,  # 창의성 (0.0 ~ 2.0)
                model_name="gpt-4o-mini",  # 모델명
                api_key = openai_api_key,
                )
                
                ai_reply = llm.invoke(f"""You are an English tutor for a {st.session_state['user_settings']['level']} learner. ### chat history ### {st.session_state["chat_history"]}""").content
                st.session_state["chat_history"].append({"role": "assistant", "content": f"""{ai_reply}"""})
            except Exception as e:
                st.error(f"Error: {e}")

    # 첨삭 버튼
    if st.button("Review My Chat"):
        user_texts = [msg["content"] for msg in st.session_state["chat_history"] if msg["role"] == "user"]
        if user_texts:
            try:
                review_response = llm.invoke(f"""You are an English tutor. Review the following conversation for grammar and clarity. ### chat history ### {st.session_state["chat_history"]}""").content
                    
                st.write("### Review and Feedback:")
                st.write(review_response)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("No user messages to review!")

