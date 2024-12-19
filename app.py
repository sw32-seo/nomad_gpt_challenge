import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler


class ChatCallbackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "AI")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.text(self.message)


@st.cache_resource(show_spinner="Embedding a file...")
def embed_file(file):
    file_content = file.read()
    file_path = f".cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir)
    splitter = CharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separator="\n",
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retreiver = vectorstore.as_retriever()
    return retreiver


def save_message(message, role):
    st.session_state['messages'].append({"content": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state['messages']:
        send_message(message['content'], message['role'], save=False)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


memory = ConversationBufferMemory(return_messages=True,
                                  memory_key="chat_history",
                                  human_prefix="human",
                                  ai_prefix="ai")


def load_memory(_):
    return memory.load_memory_variables({})["chat_history"]


prompt = ChatPromptTemplate.from_messages([("system", """
     You are a helpful assistant.
     Answer questions using ONLY the following context.
     If you don't know the answer, ust say you don't know, don't make it up.
     Context: {context}
        """), ("human", "{message}")])

st.title('Retrieval Augmented Generation')

with st.sidebar:
    api_key = st.text_input('Put your OpenAI API Key')
    github_link = ""

    file = st.file_uploader('Upload a text file', type='txt')

    app_link = "https://github.com/sw32-seo/nomad_gpt_challenge/blob/main/app.py"
    st.markdown("""
    [Github](%s)   
    """ % app_link)

if api_key and file:
    st.write('You have entered your API Key')
    llm = ChatOpenAI(model="gpt-3.5-turbo",
                     temperature=0.1,
                     api_key=api_key,
                     streaming=True,
                     callbacks=[ChatCallbackHandler()])

    retriever = embed_file(file)
    send_message("I'm ready to asnwer your questions", "AI", save=False)
    paint_history()
    message = st.chat_input("Ask me anything about your file...")

    if message:
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "message": RunnablePassthrough()
        } | prompt | llm
        send_message(message, "human")
        with st.chat_message("AI"):
            chain.invoke(message)
else:
    st.session_state['messages'] = []
