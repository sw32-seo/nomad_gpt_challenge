import os
import json
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import BaseOutputParser
from langchain.schema.runnable import RunnablePassthrough


class JsonOutputParser(BaseOutputParser):

    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

# create a folders for the cache
os.makedirs("./.cache", exist_ok=True)
os.makedirs("./.cache/quiz_files", exist_ok=True)


def format_docs(docs):
    return "\n\n".join(doc for doc in docs)


@st.cache_resource(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separator="\n",
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, difficulty):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    response = chain.invoke({"context": _docs, "difficulty": difficulty})
    return response


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(topic):
    retriever = WikipediaRetriever(top_k_results=2)
    docs = retriever.get_relevant_documents(topic)
    return docs


questions_prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant that is role playing as a teacher.

    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.

    Each question should have 4 answers, three of them must be incorrect and one should be correct.
                                                    
    Adjust the questions to the difficulty level selected by the user.

    Difficulty: {difficulty}

    Use (o) to signal the correct answer.

    Question examples:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)

    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut

    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998

    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model

    Your turn!

    Context: {context}
""")

formatting_prompt = ChatPromptTemplate.from_template("""
    You are a powerful formatting algorithm.

    You format exam questions into JSON format.
    Answers with (o) are the correct ones.

    Example Input:
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)

    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut

    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998

    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model

    Example Output:

    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }},
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }},
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }},
                ]
            }}
        ]
     }}
    ```
    Your turn!
    Questions: {context}
""")

st.set_page_config(
    page_title="QuizGPT Turbo",
    page_icon=":question:",
)

st.title("QuizGPT Turbo")

with st.sidebar:
    # Get the API key from the user in the sidebar
    api_key = st.text_input('Put your OpenAI API Key')

    # Make the user choose between a file and a Wikipedia article
    docs = None
    choice = st.selectbox("Choose what you want to use.",
                          ("File", "Wikipedia Article"))

    difficulty = st.selectbox("Choose the difficulty level of the quiz.",
                              ("Easy", "Medium", "Hard"))

    if choice == "File":
        file = st.file_uploader("Upload a file in .docx, .txt or .pdf format.",
                                type=["docx", "txt", "pdf"])
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Enter a topic.")
        if topic:
            docs = wiki_search(topic)
    if api_key:
        llm = ChatOpenAI(api_key=api_key)

        questions_chain = {
            "context": format_docs,
            "difficulty": RunnablePassthrough(),
        } | questions_prompt | llm

        formatting_chain = formatting_prompt | llm

    app_link = "https://github.com/sw32-seo/nomad_gpt_challenge/blob/main/quiz_gpt.py"
    st.markdown("""
    [Github](%s)
    """ % app_link)

if not docs:
    st.markdown("""
    Welcome to QuizGPT Turbo.

    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

    Get started by uploading a file or searching on Wikipedia in the sidebar.

    Then you can set the dificulty level of the quiz.
""")

else:
    response = run_quiz_chain(docs,
                              topic if topic else file.name,
                              difficulty=difficulty)
    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option",
                [answer["answer"] for answer in question["answers"]],
                index=None)

            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
            elif value is not None:
                st.error("Wrong!")

        button = st.form_submit_button()

        if button:
            right = 0
            for question in response["questions"]:
                if {"answer": value, "correct": True} in question["answers"]:
                    right += 1
                if right == 10:
                    st.balloons()
                    st.write("Congratulations! You got all the answers right!")
