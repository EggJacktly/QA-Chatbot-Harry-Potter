import os

import streamlit as st
import torch
from langchain import FAISS, HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from html_inputs import bot_template, user_template, css, custom_prompt_en, custom_prompt_es
import warnings

warnings.filterwarnings("ignore")

# import best model parameters
with open('./model_comparison/best_model/best-model-parameters.txt', 'r') as f:
    best_model = f.read()
f.close()


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


def get_document_chunks(documents, split_chunk_size, split_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=split_chunk_size,
        chunk_overlap=split_overlap,
        separators=["\n\n", "\n", "\t", " ", ""],
        length_function=len,
        add_start_index=True,
    )
    doc_chunks = text_splitter.split_documents(documents)
    return doc_chunks


def get_vectorstore(doc_chunks, language, new_vectorstore):
    if language == "en":
        model_name = best_model.split('\n')[2]
    elif language == "es":
        model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    else:
        raise ValueError("language must be 'en' or 'es'")

    embeddings = HuggingFaceInstructEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"}
    )
    if new_vectorstore == 'No':
        if os.path.isfile(f"data/vectorstore-{language}/index.faiss"):
            vectorstore = FAISS.load_local(
                f"data/vectorstore-{language}",
                embeddings
            )
        else:
            raise ValueError("No vectorstore found, select 'Yes' to create a new one")
    else:
        vectorstore = FAISS.from_documents(
            doc_chunks,
            embeddings
        )
        vectorstore.save_local(f"data/vectorstore-{language}")
    return vectorstore


def create_model(model_repo, language):

    if language == "en":
        bit_model = True
    elif language == "es":
        bit_model = False
    else:
        raise ValueError("language must be 'en' or 'es'")

    tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        load_in_4bit=bit_model,
        device_map='auto',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    return model, tokenizer


def get_qa_chain(vectorstore, selected_model, temperature, max_length,
                 top_p, repetition_penalty, k, custom_prompt, language):

    model_, tokenizer = create_model(selected_model, language)

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
        retriever=vectorstore.as_retriever(search_kwargs={"k": k, "search_type": "similarity"}),
        combine_docs_chain_kwargs={"prompt": custom_prompt},
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


# def clear_chat_history():
#     st.session_state.chat_history = None
#     st.session_state.conversation = None


def main():
    # load_dotenv()
    st.set_page_config(page_title='Chat with a wizard from Hogwarts 🧙‍♂️', page_icon='🧙‍♂️', layout='centered',
                       initial_sidebar_state="auto")
    st.write(css, unsafe_allow_html=True)

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
            send_key = 'Send an owl'
            header_msg = 'Chat with a wizard from Hogwarts'
            default_msg = 'Ask a question to a wizard from Hogwarts and get an answer from the Harry Potter books'
            vectorstore_msg = 'Create new vectorstore?'
            llm_msg = 'Choose a LLM model'
            # chat_history_msg = 'Clear Chat History'
            radio_yes_msg = 'Yes'
            radio_no_msg = 'No'

            custom_prompt = custom_prompt_en

            model1 = best_model.split('\n')[1]
            model2 = 'daryl149/llama-2-7b-chat-hf'

        elif selected_language == 'Spanish':
            sending_msg = 'Enviando un búho al mago...'
            language = 'es'
            send_key = "Enviar un búho"
            header_msg = 'Chatea con un mago de Hogwarts'
            default_msg = 'Haz una pregunta a un mago de Hogwarts y obtén una respuesta de los libros de Harry Potter'
            vectorstore_msg = '¿Crear nuevo vectorstore?'
            llm_msg = 'Elige un modelo LLM'
            # chat_history_msg = 'Borrar historial de chat'
            radio_yes_msg = 'Sí'
            radio_no_msg = 'No'

            custom_prompt = custom_prompt_es

            model1 = 'clibrain/Llama-2-7b-ft-instruct-es-gptq-4bit'
            model2 = 'bigscience/bloom-7b1'

        else:
            raise ValueError("language must be 'en' or 'es'")

        new_vectorstore = st.radio(vectorstore_msg, [radio_yes_msg, radio_no_msg], key='new_vectorstore')

        selected_model = st.sidebar.selectbox(llm_msg, [model1, model2],
                                              key='selected_model')

        temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0,
                                        value=float(best_model.split('\n')[4]), step=0.01)
        top_p = st.sidebar.slider('top_p', min_value=0.5, max_value=1.0, value=float(best_model.split('\n')[5]),
                                  step=0.01)
        max_length = st.sidebar.slider('max_length', min_value=512, max_value=4096, value=2048, step=512)
        k = st.sidebar.slider('k', min_value=1, max_value=10, value=5, step=1)
        repetition_penalty = st.sidebar.slider('repetition_penalty', min_value=1.0, max_value=2.0,
                                               value=float(best_model.split('\n')[6]), step=0.01)
        split_chunk_size = st.sidebar.slider('split_chunk_size', min_value=100, max_value=1000,
                                             value=int(best_model.split('\n')[7]), step=100)
        split_overlap = st.sidebar.slider('split_overlap', min_value=0, max_value=500,
                                          value=int(best_model.split('\n')[8]), step=100)

        if st.button(send_key, key='send_button'):
            with st.spinner(sending_msg):
                # loading: loading pdfs
                documents = get_raw_pdf(f"data/hp-books-{language}")

                # splitting: split pdfs to chunks
                doc_chunks = get_document_chunks(documents, split_chunk_size, split_overlap)

                # storage: create vectorstore to store chunks with embeddings
                vectorstore = get_vectorstore(doc_chunks, language, new_vectorstore)

                # chatbot: create chatbot with vectorstore
                st.session_state.conversation = get_qa_chain(vectorstore, selected_model, temperature, max_length,
                                                             top_p, repetition_penalty, k, custom_prompt, language)

    st.header(header_msg + ' 🧙‍')
    user_question = st.text_input(default_msg + ' 📚')

    if user_question:
        with st.spinner(sending_msg):
            handle_prompts(user_question)

    # st.sidebar.button(chat_history_msg, on_click=clear_chat_history)


if __name__ == "__main__":
    main()
