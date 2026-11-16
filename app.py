from flask import Flask,render_template,jsonify,request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os
from pathlib import Path

app = Flask(__name__)

project_root = Path(__file__).resolve().parent
load_dotenv(project_root / ".env")

for env_name in ("PINECONE_API_KEY", "OPENAI_API_KEY"):
    if not os.getenv(env_name):
        raise ValueError(f"{env_name} is missing from the environment or .env file.")

embeddings = download_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)


retriver = docsearch.as_retriever(search_kwargs={"k": 3},search_type="similarity")


chatModel = ChatOpenAI(model_name="gpt-4o")

question_answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(chatModel, question_answer_prompt)
rag_chain = create_retrieval_chain(retriver, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)