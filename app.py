import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template
import os

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def get_conversation_chain(vectorstore):
    try:
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",
            model_kwargs={"temperature": 0.5, "max_length": 512}
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process PDF documents first!")
        return
    
    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def main():
    load_dotenv()
    st.set_page_config(
        page_title="PDFSense", 
        page_icon="📚",
        layout="wide"
    )
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("PDFSense 📚")

    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        st.warning("⚠️ Please set your HUGGINGFACEHUB_API_TOKEN in your .env file")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDF Documents", 
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if pdf_docs:
            st.success(f"Uploaded {len(pdf_docs)} PDF(s)")
            
        if st.button("🔄 Process Documents"):
            if not pdf_docs:
                st.error("Please upload at least one PDF document!")
            else:
                with st.spinner("Processing documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    
                    if not raw_text.strip():
                        st.error("No text could be extracted from the PDFs!")
                        return
                    
                    text_chunks = get_text_chunks(raw_text)
                    st.info(f"Created {len(text_chunks)} text chunks")
                    
                    vectorstore = get_vectorstore(text_chunks)
                    
                    if vectorstore:
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                        
                        if st.session_state.conversation:
                            st.success("✅ Documents processed successfully! You can now ask questions.")
                        else:
                            st.error("Failed to create conversation chain")
                    else:
                        st.error("Failed to create vector store")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = None
            st.success("Chat history cleared!")
        
        if st.session_state.conversation:
            st.success("✅ Ready to chat!")
        else:
            st.info("Upload and process PDFs to start chatting")

if __name__ == "__main__":
    main()