import streamlit as st
import fitz
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.create_documents([text])

def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore

def get_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.environ.get("GROQ_API_KEY")
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate.from_template("""
Use the following context to answer the question.
If you don't know the answer, say "I don't know".

Context: {context}

Question: {question}

Answer:""")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

st.title("📄 RAG Research Agent")
st.caption("Upload PDFs and ask questions about them")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    if "vectorstore" not in st.session_state:
        all_text = ""
        with st.spinner("Reading PDFs..."):
            for f in uploaded_files:
                all_text += extract_text(f)

        with st.spinner("Creating embeddings... (first time takes ~30 seconds)"):
            chunks = get_chunks(all_text)
            st.session_state.vectorstore = get_vectorstore(chunks)
            st.session_state.chain, st.session_state.retriever = get_chain(st.session_state.vectorstore)

        st.success(f"✅ {len(chunks)} chunks ready! Ask anything below.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about your documents..."):
    if "chain" not in st.session_state:
        st.warning("⚠️ Please upload a PDF first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(prompt)
                sources = st.session_state.retriever.invoke(prompt)

            st.write(answer)

            with st.expander("📚 Source chunks used"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)

        st.session_state.messages.append({"role": "assistant", "content": answer})