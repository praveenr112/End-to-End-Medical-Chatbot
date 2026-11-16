from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

# Extract text from PDF files.

def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def filter_to_mimimal_docs(documents: List[Document]) -> List[Document]:
    """
    Filter the documents to only include those with a page_content length greater than 100.
    """
    minimal_docs:List[Document] = []
    for doc in documents:
        src = doc.metadata.get("source")
        minimal_docs.append(Document(page_content=doc.page_content, metadata={"source": src}))
        
    return minimal_docs


def text_split(minimal_documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunk = text_splitter.split_documents(minimal_documents)
    return text_chunk

def download_embeddings():
    """
    Download and return the Hugging Face embeddings.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
