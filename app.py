import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
import asyncio
import shutil

# Setup
st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚", layout="wide")
load_dotenv()

# Async fix
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Utility
def process_pdfs(files):
    text = ""
    for f in files:
        reader = PdfReader(f)
        for p in reader.pages:
            if p.extract_text():
                text += p.extract_text()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200).split_text(text)
    vs = FAISS.from_texts(chunks, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    vs.save_local("faiss_index")
    return vs

def load_vs():
    return FAISS.load_local("faiss_index",
                            GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
                            allow_dangerous_deserialization=True)

def make_chain(vs):
    return ConversationalRetrievalChain.from_llm(
        llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3),
        retriever=vs.as_retriever(),
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer"),
        output_key="answer"
    )

# State
if "chain" not in st.session_state:
    if os.path.exists("faiss_index"):
        st.session_state.chain = make_chain(load_vs())
    else:
        st.session_state.chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending" not in st.session_state:
    st.session_state.pending = None  # assistant typing placeholder

# Sidebar UI
with st.sidebar:
    st.header("📂 Upload PDFs")
    pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True, type="pdf")

    if st.button("Process PDFs"):
        if pdf_docs:
            with st.spinner("Processing PDFs..."):
                vs = process_pdfs(pdf_docs)
                st.session_state.chain = make_chain(vs)
                st.success("✅ PDFs processed. Start chatting!")
        else:
            st.error("⚠️ Upload at least one PDF.")

    if st.button("Clear Chat"):
        st.session_state.messages.clear()
        st.session_state.pending = None
        st.success("🧹 Chat cleared.")
    
    if st.button("Reset PDFs"):
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")
        st.session_state.chain = None
        st.session_state.messages.clear()
        st.session_state.pending = None
        st.success("🔄 PDFs reset.")

    if st.session_state.chain:
        st.markdown("**✅ Ready to Chat**")
    else:
        st.markdown("**⚠ Upload & index PDFs to start**")

st.title("🤖 Chat with your PDFs")

# ---- Custom CSS ----
st.markdown("""
<style>
:root {
    --user-light: linear-gradient(135deg, #228be6, #4dabf7);
    --user-dark: linear-gradient(135deg, #1c7ed6, #339af0);
    --bot-light: #f8f9fa;
    --bot-dark: #2b2b2b;
    --text-light: #212529;
    --text-dark: #f1f3f5;
}

/* Wider layout */
section.main > div {
    max-width: 1100px !important;
    margin: auto;
    padding: 1rem 2rem 6rem 2rem;
}

/* Scrollable chat */
.chat-box {
    max-height: 70vh;
    overflow-y: auto;
    padding-right: 10px;
}

/* Chat bubbles */
.msg {
    padding: 12px 16px;
    border-radius: 12px;
    margin: 8px;
    max-width: 50%;
    font-size: 18px;
    line-height: 1.5;
    word-wrap: break-word;
    animation: fadeInUp 0.3s ease forwards;
}

.user { margin-left: auto; text-align: right; }
.bot { margin-right: auto; text-align: left; }

@media (prefers-color-scheme: dark) {
    .user { background: var(--user-dark); color: var(--text-dark); }
    .bot { background: var(--bot-dark); color: var(--text-dark); border: 1px solid #444; }
}
@media (prefers-color-scheme: light) {
    .user { background: var(--user-light); color: white; }
    .bot { background: var(--bot-light); color: var(--text-light); border: 1px solid #ddd; }
}

/* Typing bubble */
.typing {
    background: #444;
    color: #eee;
    font-style: italic;
    margin-right: auto;
    border-radius: 12px;
    padding: 10px 14px;
    display: inline-flex;
    align-items: center;
}
.dot {
    height: 8px; width: 8px;
    margin: 0 2px;
    background-color: currentColor;
    border-radius: 50%;
    display: inline-block;
    animation: blink 1.4s infinite both;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes blink { 0%,100%{opacity:.2;} 20%{opacity:1;} }
@keyframes fadeInUp { 0%{opacity:0;transform:translateY(10px);} 100%{opacity:1;transform:translateY(0);} }

/* Pin input box */
.stChatInput {
    position: fixed !important;
    bottom: 15px;
    left: 50%;
    transform: translateX(-50%);
    width: 950px !important;
    max-width: 95%;
    z-index: 999;
}
</style>
""", unsafe_allow_html=True)

# ---- Chat UI ----
with st.container():
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    for msg in st.session_state.messages:
        cls = "user" if msg["role"] == "user" else "bot"
        st.markdown(f"<div class='msg {cls}'>{msg['content']}</div>", unsafe_allow_html=True)

    if st.session_state.pending:
        st.markdown(
            "<div class='typing'>Assistant is typing <span class='dot'></span><span class='dot'></span><span class='dot'></span></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Input ----
if user_in := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": user_in})
    st.session_state.pending = True
    st.rerun()

# ---- Response ----
if st.session_state.pending:
    if st.session_state.chain:
        resp = st.session_state.chain.invoke({"question": st.session_state.messages[-1]["content"]})["answer"]
    else:
        resp = "⚠ Please upload and index PDFs first."
    st.session_state.messages.append({"role": "assistant", "content": resp})
    st.session_state.pending = None
    st.rerun()
