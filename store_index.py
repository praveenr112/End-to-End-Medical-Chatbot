import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import (
    download_embeddings,
    filter_to_mimimal_docs,
    load_pdf_files,
    text_split,
)

project_root = Path(__file__).resolve().parent
load_dotenv(project_root / ".env")

pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is missing from the environment or .env file.")

data_dir = project_root / "data"
if not data_dir.is_dir():
    raise FileNotFoundError(f"PDF data directory not found: {data_dir}")

extracted_data = load_pdf_files(data=str(data_dir))
if not extracted_data:
    raise FileNotFoundError(f"No PDF files found in {data_dir}")

filter_data = filter_to_mimimal_docs(extracted_data)
text_chunk = text_split(filter_data)
if not text_chunk:
    raise ValueError(f"No text could be extracted from PDF files in {data_dir}")

PC = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"

if not PC.has_index(index_name):
    PC.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

embeddings = download_embeddings()
PineconeVectorStore.from_documents(
    documents=text_chunk,
    embedding=embeddings,
    index_name=index_name
)
