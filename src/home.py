import streamlit as st

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain.chains.llm import LLMChain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate

import styles

# Streamlit UI
st.title("🤖 Local RAG System with Ollama & Deepseek R1")

st.header("📄 Upload your PDF file")
uploaded_file = st.file_uploader("upload here", type="pdf")

if uploaded_file is not None:
    st.success("PDF uploaded successfully! Processing...")

    # Save the uploaded file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Load the PDF
    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()

    # Split the document into chunks
    st.subheader("📚 Splitting the document into chunks...")
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)

    # Instantiate the embedding model
    embedder = HuggingFaceEmbeddings()

    # Create vector store and retriever
    st.subheader("🔍 Creating embeddings and setting up the retriever...")
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Define the LLM and the prompt
    llm = Ollama(model="deepseek-r1:1.5b")
    prompt = """
    1. Use the following pieces of context to answer the question at the end.
    2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
    3. Keep the answer crisp and limited to 3,4 sentences.
    Context: {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    # Define the document and combination chains
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, verbose=True)
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\ncontent:{page_content}\nsource:{source}",
    )
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
        verbose=True
    )

    qa = create_retrieval_chain(
        combine_documents_chain=combine_documents_chain,
        retriever=retriever,
        verbose=True,
        return_source_documents=True
    )

    # Question input and response display
    st.header("❓ Ask a Question")
    user_input = st.text_input("Type your question related to the document:")

    if user_input:
        with st.spinner("Processing your query..."):
            try:
                response = qa(user_input)["result"]
                st.success("✅ Response:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a PDF file to start.")
