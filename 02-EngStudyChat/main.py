import streamlit as st
import openai

# Streamlit 애플리케이션 설정
st.set_page_config(page_title="AI English Chat", layout="wide")

# 사이드바: API Key 입력
st.sidebar.title("AI English Chat")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
if not api_key:
    st.sidebar.warning("Please enter your API Key to start.")
    st.stop()

# OpenAI API 설정
openai.api_key = api_key

# 사용자 설정 변수 초기화
if "user_settings" not in st.session_state:
    st.session_state["user_settings"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

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
        st.experimental_rerun()

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
    
    # 사용자 입력
    user_input = st.text_input("Your message", "")
    if st.button("Send"):
        if user_input.strip():
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            
            # OpenAI API 호출
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": f"You are an English tutor for a {st.session_state['user_settings']['level']} learner."},
                        *st.session_state["chat_history"]
                    ],
                    max_tokens=200,
                    temperature=0.7
                )
                ai_reply = response["choices"][0]["message"]["content"]
                st.session_state["chat_history"].append({"role": "assistant", "content": ai_reply})
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # 첨삭 버튼
    if st.button("Review My Chat"):
        user_texts = [msg["content"] for msg in st.session_state["chat_history"] if msg["role"] == "user"]
        if user_texts:
            try:
                review_response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an English tutor. Review the following conversation for grammar and clarity."},
                        {"role": "user", "content": "\n".join(user_texts)}
                    ],
                    max_tokens=300,
                    temperature=0.5
                )
                st.write("### Review and Feedback:")
                st.write(review_response["choices"][0]["message"]["content"])
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("No user messages to review!")

