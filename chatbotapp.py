import os

import streamlit as st
from dotenv import load_dotenv
from langchain import FAISS, HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_raw_pdf(pdfs_path):
    loader = DirectoryLoader(
        pdfs_path,
        glob="./*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    return loader.load()


def get_document_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # split_chunk_size
        chunk_overlap=200,  # split_overlap
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)


def get_vectorstore(doc_chunks, language):
    # language = "en" or "es"
    embeddings = HuggingFaceInstructEmbeddings(model_name="BAAI/bge-large-en")
    if os.path.isfile(f"data/vectorstore_{language}/index.faiss"):
        vectorstore = FAISS.load_local(f"data/vectorstore_{language}")
    else:
        vectorstore = FAISS.from_documents(
            doc_chunks,
            embeddings
        )
        vectorstore.save_local(f"data/vectorstore_{language}")
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",
                         model_kwargs={"temperature": 0, "max_length": 1024})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with a wizard from Hogwarts", page_icon=":wizard:", layout="centered",
                       initial_sidebar_state="auto")
    st.header("Chat with a wizard from Hogwarts :wizard:")
    st.text_input("Ask a question to a wizard from Hogwarts")

    with st.sidebar:
        st.subheader("Which language do you speak? \n ¿Qué idioma hablas?")
        if st.button("English - Inglés"):
            with st.spinner("Sending an owl to the wizard..."):
                # loading: loading pdfs
                documents = get_raw_pdf("data/hp-books-english")

                # splitting: split pdfs to chunks
                doc_chunks = get_document_chunks(documents)

                # storage: create vectorstore to store chunks with embeddings
                vectorstore = get_vectorstore(doc_chunks, language="en")

                # chatbot: create chatbot with vectorstore
                st.session_state.conversation = get_conversation_chain(vectorstore)

        if st.button("Español - Spanish"):
            with st.spinner("Enviando un búho al mago..."):
                # loading: loading pdfs
                documents = get_raw_pdf("data/hp-books-spanish")

                # splitting: split pdfs to chunks
                doc_chunks = get_document_chunks(documents)

                # storage: create vectorstore to store chunks with embeddings
                vectorstore = get_vectorstore(doc_chunks, language="es")

                # chatbot: create chatbot with vectorstore
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
