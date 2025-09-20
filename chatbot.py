import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# Get Google Generative AI API key from environment variables
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")

os.environ["GOOGLE_API_KEY"] = api_key


text_content = """Machine learning is a subset of artificial intelligence (AI) 
that enables systems to learn from data and make decisions without explicit programming. 
It can be categorized into three main types: supervised learning, unsupervised learning, 
and reinforcement learning."""

# Save text content to a file
with open("machine_learning.txt", "w") as file:
    file.write(text_content)

print("✅ Sample text file 'machine_learning.txt' created successfully!")

# Load the text file using LangChain's TextLoader
loader = TextLoader("machine_learning.txt") # Load the text file
documents = loader.load()                   # Converts the file into a list of document objects

# Split the document into smaller chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,         # Each chunk will have a maximum of 500 characters
    chunk_overlap=50        # Overlap of 50 characters between chunks
)

docs = text_splitter.split_documents(documents)

# Create embeddings and store them in a FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings) # Store embeddings in FAISS

# Load Google Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

# Create a RetrievalQA chain (Connects FAISS retriever with the LLM)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

print("\n🚀 Gemini 2.0 RAG Chatbot is ready! Ask a question about Machine Learning.")

# Step 9: Interactive chatbot loop
while True:
    query = input("\nAsk a question (or type 'exit' to quit): ")
    
    if query.lower() == "exit":   # Exit condition
        print("👋 Goodbye!")
        break

    # Run the query through the RAG pipeline
    answer = qa.run(query)
    
    # Print the AI's answer
    print("\n🤖 AI:", answer)
