import os

import streamlit as st
import torch
from dotenv import load_dotenv
from langchain import FAISS, HuggingFaceHub, HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from html_inputs import user_template, bot_template


def get_raw_pdf(pdfs_path):
    loader = DirectoryLoader(
        pdfs_path,
        loader_cls=PyPDFLoader,
        glob="./*.pdf",
        show_progress=True,
        use_multithreading=True
    )
    documents = loader.load()
    return documents


def get_document_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # split_chunk_size
        chunk_overlap=200,  # split_overlap
        length_function=len,
        add_start_index=True,
    )
    doc_chunks = text_splitter.split_documents(documents)
    return doc_chunks


def get_vectorstore(doc_chunks, language):
    if language == "en":
        model_name = "BAAI/bge-large-en"
    elif language == "es":
        model_name = "intfloat/multilingual-e5-large"
    else:
        raise ValueError("language must be 'en' or 'es'")

    embeddings = HuggingFaceInstructEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"}
    )

    if os.path.isfile(f"data/vectorstore_{language}/index.faiss"):
        vectorstore = FAISS.load_local(
            f"data/vectorstore_{language}",
            embeddings
        )
    else:
        vectorstore = FAISS.from_documents(
            doc_chunks,
            embeddings
        )
        vectorstore.save_local(f"data/vectorstore_{language}")
    return vectorstore


def create_model(model_repo):
    tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=True)
    # model = AutoModelForCausalLM.from_pretrained(
    model = AutoModelForConditionalGeneration.from_pretrained(
        model_repo,
        load_in_4bit=True,
        device_map='auto',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    return model, tokenizer


def get_qa_chain(vectorstore, selected_model, temperature, max_length, top_p, repetition_penalty=1.15):
    # llm = HuggingFaceHub(repo_id=selected_model,
    #                      model_kwargs={"temperature": temperature,
    #                                    "max_length": max_length,
    #                                    "top_p": top_p}
    #                      )

    model_, tokenizer = create_model(selected_model)

    pipe = pipeline(
        task="text-generation",
        model=model_,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return qa_chain


def handle_prompts(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def clear_chat_history():
    st.experimental_rerun()
    # st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


def main():
    # load_dotenv()
    st.set_page_config(page_title='Chat with a wizard from Hogwarts 🧙‍♂️', page_icon='🧙‍♂️', layout='centered',
                       initial_sidebar_state="auto")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    with st.sidebar:
        st.subheader('Models and parameters')
        selected_language = st.sidebar.selectbox('Choose a language / Elige un idioma',
                                                 ['English', 'Spanish'],
                                                 key='selected_language')
        if selected_language == 'English':
            language = 'en'
            sending_msg = 'Sending an owl to the wizard...'
            send_key = "Send"
            header_msg = 'Chat with a wizard from Hogwarts'
            default_msg = 'Ask a question to a wizard from Hogwarts and get an answer from the Harry Potter books'

            model1 = 'google/flan-t5-xxl'
            model2 = 'daryl149/llama-2-7b-chat-hf'

        elif selected_language == 'Spanish':
            sending_msg = "Enviando un búho al mago..."
            language = 'es'
            send_key = "Enviar"
            header_msg = 'Chatea con un mago de Hogwarts'
            default_msg = 'Haz una pregunta a un mago de Hogwarts y obtén una respuesta de los libros de Harry Potter'

            model1 = 'tiiuae/falcon-180B-chat'
            model2 = 'bigscience/bloom'

        else:
            raise ValueError("language must be 'en' or 'es'")

        selected_model = st.sidebar.selectbox('Choose a LLM model', [model1, model2],
                                              key='selected_model')

        temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
        top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.95, step=0.01)
        max_length = st.sidebar.slider('max_length', min_value=512, max_value=4096, value=2048, step=512)

        if st.button(send_key, key='send_button'):
            with st.spinner(sending_msg):
                # loading: loading pdfs
                documents = get_raw_pdf(f"data/hp-books-{language}")

                # splitting: split pdfs to chunks
                doc_chunks = get_document_chunks(documents)

                # storage: create vectorstore to store chunks with embeddings
                vectorstore = get_vectorstore(doc_chunks, language)

                # chatbot: create chatbot with vectorstore
                st.session_state.conversation = get_qa_chain(vectorstore, selected_model, temperature, max_length,
                                                             top_p)

    st.header(header_msg + ' 🧙‍')
    user_question = st.text_input(default_msg + ' 📚')

    if user_question:
        with st.spinner(sending_msg):
            handle_prompts(user_question)

    # # Display or clear chat messages
    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.write(message["content"])

    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


if __name__ == "__main__":
    main()
