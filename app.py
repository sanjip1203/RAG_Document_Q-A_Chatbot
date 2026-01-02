import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain



# ------------------ ENV ------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment (.env).")
    st.stop()

# ------------------ LLM ------------------


llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",  
    temperature=0
)


# ------------------ PROMPT ------------------
prompt = ChatPromptTemplate.from_template(
    """
Answer the question based ONLY on the provided context.
If the answer is not in the context, say: "I don't know based on the provided documents."

<context>
{context}
</context>

Question: {input}
"""
)

# ------------------ VECTOR BUILD ------------------
def create_vector_embeddings():
    if "vectors" in st.session_state:
        return

    st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")

    loader = PyPDFDirectoryLoader("research_papers")
    docs = loader.load()

    st.write(" PDFs loaded:", len(docs))
    if len(docs) == 0:
        st.error(" No PDFs found in research_papers/")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    final_docs = splitter.split_documents(docs)   
    st.write(" Chunks made:", len(final_docs))  

    st.session_state.vectors = FAISS.from_documents(
        final_docs,
        st.session_state.embeddings
    )

    st.write(" FAISS index size:", st.session_state.vectors.index.ntotal)

# ------------------ UI ------------------
st.title("Research Paper Q&A (RAG)")

user_prompt = st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embeddings()
    st.success("Vector Database is ready ")

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Click **Document Embedding** first to build the vector DB.")
        st.stop()

    retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 4})

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    st.caption(f"Response time: {time.process_time() - start:.3f}s")

    st.subheader("Answer")
    st.write(response["answer"])

    with st.expander("Document similarity search"):
        for i, doc in enumerate(response["context"], start=1):
            st.markdown(f"**Chunk {i}**")
            st.write(doc.page_content)
            st.write("-" * 50)
