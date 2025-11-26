import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

GEMINI_API_KEY = "xxxyyyyyzzzz"
if not GEMINI_API_KEY:
    st.error("Please set GEMINI_API_KEY environment variable (export GEMINI_API_KEY=...)")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)


# Gemini Embeddings wrapper
class GeminiEmbeddings:
    def __init__(self, model="models/embedding-001"):
        self.model = model

    def _embed_single(self, text: str):
        # genai.embed_content returns a mapping with 'embedding' key
        res = genai.embed_content(model=self.model, content=text)
        # different client versions may return slightly different shapes
        if isinstance(res, dict) and "embedding" in res:
            return res["embedding"]
        # fallback attempt for nested shapes
        try:
            return res["data"][0]["embedding"]
        except Exception:
            # raise a helpful error if embedding cannot be parsed
            raise RuntimeError(f"Unexpected embeddings response shape: {res}")

    def embed_documents(self, texts):
        return [self._embed_single(t) for t in texts]

    def embed_query(self, text):
        return self._embed_single(text)



# Creating streamlit UI 
st.header("QueryCHAT")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF & ask questions in QueryCHAT", type="pdf")

if not file:
    st.info("Upload a PDF to get started.")
    st.stop()

# Read PDF using PyPDF2
pdf_reader = PdfReader(file)
text = ""
for page in pdf_reader.pages:
    page_text = page.extract_text() or ""
    text += page_text + "\n"

# Split the data(PDF) into chunks
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)
chunks = text_splitter.split_text(text)

# Create+ embeddings object
embeddings = GeminiEmbeddings(model="models/embedding-001")

# Create FAISS index robustly across langchain versions
vector_store = None
errors = []

# Try several constructor signatures used across langchain versions
constructors = [
    lambda: FAISS.from_texts(chunks, embeddings),            # older versions
    lambda: FAISS.from_texts(chunks, embedding=embeddings),  # some expect keyword 'embedding'
    lambda: FAISS.from_texts(chunks, embedding_function=embeddings.embed_query),  # direct callable
    lambda: FAISS.from_texts(chunks, embedding_function=embeddings),  # sometimes expects callable/object
]

for c in constructors:
    try:
        vector_store = c()
        break
    except Exception as e:
        errors.append(e)

if vector_store is None:
    # If none worked, raise an error with collected traces
    st.error("Failed to create FAISS vector store. Tried multiple constructor signatures.")
    st.write("Errors encountered (first lines):")
    for i, e in enumerate(errors[:4], 1):
        st.write(f"{i}. {type(e).__name__}: {str(e)[:300]}")
    st.stop()

# Ensure vector_store has a callable embedding function
# Some versions set vector_store.embedding_function to the embeddings object (non-callable)
if hasattr(vector_store, "embedding_function"):
    if not callable(vector_store.embedding_function):
        # If the embeddings object has embed_query, use that
        if hasattr(embeddings, "embed_query") and callable(embeddings.embed_query):
            vector_store.embedding_function = embeddings.embed_query
        elif hasattr(embeddings, "embed_documents") and callable(embeddings.embed_documents):
            # wrap embed_documents to return a single vector for a single text
            vector_store.embedding_function = lambda text: embeddings.embed_documents([text])[0]
        else:
            st.error("Cannot set embedding function: embeddings wrapper has no usable method.")
            st.stop()
else:
    # older/newer builds might use _embed_query or similar — set a safe attribute too
    vector_store.embedding_function = embeddings.embed_query

# Now user input -> search -> generate
user_question = st.text_input("Ask a question from the uploaded PDF")

if user_question:
    k = 4
    try:
        # prefer similarity_search_with_score if you want scores; similarity_search returns docs
        docs = vector_store.similarity_search(user_question, k=k)
    except TypeError:
        # fallback signature (some versions expect positional k)
        docs = vector_store.similarity_search(user_question, k)

    # Ensure docs is a list of docs (some versions return (doc, score))
    # If docs are (doc, score) pairs, normalize:
    if len(docs) > 0 and isinstance(docs[0], tuple):
        docs = [d for d, s in docs]

    context = "\n\n".join([d.page_content for d in docs])

    prompt = (
        "You are a helpful assistant that answers questions strictly based on the provided context from the PDF document. "
        "Only answer questions using information from the context below. "
        "If the question cannot be answered using the context, respond with: "
        "'I can only answer questions related to the uploaded PDF document.'\n\n"
        f"Context:\n{context}\n\nHuman: {user_question}\nAssistant:"
    )


    

    # Call Gemini model
    try:
        gen = genai.GenerativeModel("gemini-2.0-flash")
        gen_resp = gen.generate_content(prompt)
        # try common shapes for the response text
        answer = getattr(gen_resp, "text", None)
        if not answer:
            try:
                answer = gen_resp.output[0].contents[0].text
            except Exception:
                answer = str(gen_resp)
    except Exception as e:
        st.error(f"Error calling Gemini: {e}")
        answer = None

    if answer:
        st.write(answer)
    else:
        st.write("No answer returned. Check API key, model availability, or inspect logs.")







