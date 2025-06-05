Here is the **complete README in markdown format**:

````markdown
# 📚 PDFSense — Intelligent PDF Chatbot using LLMs

PDFSense is a powerful Streamlit-based web app that lets you upload one or more PDF documents and interact with them conversationally using a Large Language Model (LLM). The app uses Hugging Face's `Flan-T5` model to answer questions based on the uploaded PDF content.

---

## 🔍 Features

- 📄 Upload multiple PDF documents
- ✂️ Extract and chunk text intelligently
- 🤖 Ask natural language questions about the documents
- 🧠 Context-aware conversation with memory
- 🔄 Clear chat history and restart conversation anytime
- 🌐 Powered by Hugging Face Transformers and LangChain

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **PDF Parsing**: PyPDF2
- **LLM**: [Flan-T5-Large](https://huggingface.co/google/flan-t5-large) via Hugging Face Hub
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: FAISS (for similarity search)
- **Conversational Chain**: LangChain `ConversationalRetrievalChain`

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pdfsense.git
cd pdfsense
````

### 2. Install Dependencies

Use `pip` to install the required packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, create one with:

```txt
streamlit
python-dotenv
PyPDF2
langchain
faiss-cpu
sentence-transformers
huggingface-hub
```

### 3. Set Up Environment Variables

Create a `.env` file in the root directory and add your Hugging Face API token:

```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

Get your token from: [Hugging Face Settings](https://huggingface.co/settings/tokens)

### 4. Run the App

```bash
streamlit run app.py
```

---

## 🧪 How It Works

1. **Upload PDFs**: Upload one or more PDF files via the sidebar.
2. **Text Extraction**: Uses `PyPDF2` to extract raw text from each page.
3. **Text Chunking**: Text is split into overlapping chunks using LangChain's `CharacterTextSplitter`.
4. **Embedding**: Each chunk is embedded using `all-MiniLM-L6-v2`.
5. **Vector Store**: FAISS stores embeddings for fast similarity search.
6. **Conversational Retrieval**: Ask questions; relevant chunks are retrieved and passed to `Flan-T5` to generate a response.

---

## ⚠️ Known Issues

* Image-based (scanned) PDFs are not supported yet.
* Free-tier Hugging Face API tokens may face rate limits.

---

## ✅ To Do

* [ ] Add OCR support for scanned PDFs
* [ ] Enable model selection via UI
* [ ] Add multi-language support
* [ ] Save chat history persistently


