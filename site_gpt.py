import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

answers_prompt = ChatPromptTemplate.from_template("""
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always follow the format of examples and include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
""")


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     answer = answers_chain.invoke({
    #         "context": doc.page_content,
    #         "question": question
    #     })
    #     answers.append(answer.content)
    return {
        "question":
        question,
        "answers": [{
            "answer":
            answers_chain.invoke({
                "context": doc.page_content,
                "question": question
            }).content,
            "source":
            doc.metadata["source"],
            "date":
            doc.metadata['lastmod']
        } for doc in docs]
    }


choose_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
            Use ONLY the following pre-existing answers to answer the user's question.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.
            Cite sources and return the sources of the answers as they are, do not change them.
            Answers: {answers}
            """,
    ),
    ("human", "{question}"),
])


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join([
        f"Answer: {answer['answer']}\nSource: {answer['source']}\nDate: {answer['date']}"
        for answer in answers
    ])
    return choose_chain.invoke({"answers": condensed, "question": question})


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").strip()


@st.cache_resource(show_spinner="Loading website...")
def load_website(url: str):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/ai-gateway\/.*)$",  # only ai-gateway pages
            r"^(.*\/vectorize\/.*)$",  # only vectorize pages
            r"^(.*\/workers-ai\/.*)$",  # only workers-ai pages
        ],
        parsing_function=parse_page)
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs,
                                        OpenAIEmbeddings(api_key=api_key))
    return vector_store.as_retriever()


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🌐",
)

st.title("SiteGPT")

html2text_transformer = Html2TextTransformer()

st.write(
    "Ask questions about the content of a website. \n Start by write the URL of the website on the sidebar."
)

with st.sidebar:
    api_key = st.text_input("OpenAI API Key")
    url = "https://developers.cloudflare.com/sitemap-0.xml"

    app_link = "https://github.com/sw32-seo/nomad_gpt_challenge/blob/main/site_gpt.py"
    st.markdown("""
    [Github](%s)
    """ % app_link)

if api_key:
    llm = ChatOpenAI(temperature=0.1, api_key=api_key)
    retriever = load_website(url)
    query = st.text_input("Ask a question to the website")
    if query:
        chain = {
            "docs": retriever,
            "question": RunnablePassthrough(),
        } | RunnableLambda(get_answers) | RunnableLambda(choose_answer)

        result = chain.invoke(query)
        st.write(result.content)
