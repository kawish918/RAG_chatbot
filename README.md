# 🤖 RAG Chatbot with Google Gemini 2.0

A **Retrieval-Augmented Generation (RAG)** chatbot built with LangChain, HuggingFace embeddings, FAISS vector store, and Google's Gemini 2.0 Flash model. This chatbot can answer questions by retrieving relevant information from your documents and generating contextual responses.

## ✨ Features

- **Document Loading**: Automatically loads and processes text documents
- **Text Chunking**: Intelligently splits documents into manageable chunks
- **Vector Embeddings**: Uses HuggingFace sentence transformers for semantic search
- **FAISS Vector Store**: Fast similarity search and clustering of dense vectors
- **Google Gemini 2.0**: Powered by Google's latest generative AI model
- **Interactive Chat**: Real-time question-answering interface
- **RAG Pipeline**: Combines retrieval and generation for accurate, context-aware responses

## 📋 Prerequisites

- Python 3.8 or higher
- Google AI Studio API key
- Windows/Linux/macOS

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/kawish918/RAG_chatbot.git
cd rag_chatbot
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv rag_env

# Activate virtual environment
# On Windows:
rag_env\Scripts\activate
# On macOS/Linux:
source rag_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install langchain-community
pip install langchain
pip install langchain-huggingface
pip install faiss-cpu
pip install langchain-google-genai
pip install sentence-transformers
```

### 4. Get Google AI API Key

1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the API key

### 5. Configure API Key

1. Copy the example environment file and create your own `.env` file:

```bash
# Copy the template
cp .env.example .env
```

2. Edit the `.env` file and add your actual Google API key:

```env
GOOGLE_API_KEY=your_actual_api_key_here
```

### 6. Run the Chatbot

```bash
python chatbot.py
```

## 💡 How It Works

### RAG Architecture

1. **Document Loading**: The system loads text documents using LangChain's TextLoader
2. **Text Splitting**: Documents are split into smaller chunks (500 characters with 50-character overlap)
3. **Embedding Generation**: Each chunk is converted to vector embeddings using HuggingFace's `all-MiniLM-L6-v2` model
4. **Vector Storage**: Embeddings are stored in FAISS for fast similarity search
5. **Query Processing**: User questions are embedded and matched against stored vectors
6. **Context Retrieval**: Most relevant document chunks are retrieved
7. **Response Generation**: Google Gemini 2.0 generates answers based on retrieved context

### Code Structure

```
rag_chatbot/
├── chatbot.py              # Main chatbot application
├── machine_learning.txt    # Sample knowledge base
├── README.md              # This file
└── rag_env/               # Virtual environment
```

## 🔧 Configuration

### Customizing Text Chunking

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,         # Adjust chunk size
    chunk_overlap=50        # Adjust overlap
)
```

### Changing Embedding Model

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Change model
)
```

### Using Different LLM Models

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")  # Available models
```

## 📚 Adding Your Own Documents

1. Replace or add content to `machine_learning.txt`
2. Or modify the code to load multiple documents:

```python
# Load multiple files
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("./documents", glob="*.txt")
documents = loader.load()
```

##  Example Usage

```
🚀 Gemini 2.0 RAG Chatbot is ready! Ask a question about Machine Learning.

Ask a question (or type 'exit' to quit): What is machine learning?

🤖 AI: Machine learning is a subset of artificial intelligence (AI) that enables 
systems to learn from data and make decisions without explicit programming. It allows 
computers to automatically improve their performance on a specific task through 
experience, without being explicitly programmed for every possible scenario.

Ask a question (or type 'exit' to quit): What are the types of machine learning?

🤖 AI: Based on the information provided, machine learning can be categorized into 
three main types:

1. **Supervised Learning** - Learning with labeled data
2. **Unsupervised Learning** - Learning patterns from unlabeled data  
3. **Reinforcement Learning** - Learning through interaction and feedback

Ask a question (or type 'exit' to quit): exit
👋 Goodbye!
```

## 🛠️ Troubleshooting

### Common Issues

**1. API Key Error**
```
Error: Invalid API key
```
- Solution: Ensure your Google AI API key is correct and active

**2. Import Errors**
```
ModuleNotFoundError: No module named 'langchain'
```
- Solution: Activate virtual environment and install dependencies

**3. FAISS Installation Issues**
```
Error installing faiss-cpu
```
- Solution: Use `pip install faiss-cpu` or `conda install faiss-cpu`


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [HuggingFace](https://huggingface.co/) for embedding models
- [Google AI](https://ai.google/) for Gemini 2.0 model
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search


