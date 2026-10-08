import streamlit as st
import fitz
import ollama
import chromadb
import os
import requests
from bs4 import BeautifulSoup
import mysql.connector

# ── Page Config ────────────────────────
st.set_page_config(
    page_title="Local RAG System",
    page_icon="🤖",
    layout="wide"
)

# ── ChromaDB Setup ─────────────────────
if "collection" not in st.session_state:
    st.session_state.client = chromadb.PersistentClient(path="./chroma_db")
    st.session_state.collection = st.session_state.client.get_or_create_collection(
        name="pdf_chunks",
        metadata={"hnsw:space": "cosine"}
    )
client = st.session_state.client
collection = st.session_state.collection

# ── Functions ──────────────────────────

def load_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def load_from_url(url):
    
    # Fetch webpage
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    
    # Check success
    if response.status_code != 200:
        print(f"Failed! Status: {response.status_code}")
        return ""
    
    # Parse HTML
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Remove unwanted tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    
    # Extract clean text
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)
    
    print(f"URL loaded ✅ ({len(clean_text)} characters)")
    return clean_text

def load_from_mysql(host, user, password, database):
    """Load ALL tables from MySQL database"""
    try:
        conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        cursor = conn.cursor()
        
        # Get ALL tables automatically
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        
        full_text = ""
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            full_text += f"Table: {table_name}\n"
            full_text += f"Columns: {', '.join(columns)}\n"
            
            for row in rows:
                row_text = ", ".join([
                    f"{col}: {val}"
                    for col, val in zip(columns, row)
                ])
                full_text += row_text + "\n"
            full_text += "\n"
        
        cursor.close()
        conn.close()
        return full_text
    
    except Exception as e:
        return f"Error: {str(e)}"




def split_into_chunks(full_text, chunk_size=200, overlap=20):
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def generate_embedding(text):
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    return response["embedding"]

def generate_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=chunk
        )
        embeddings.append(response["embedding"])
    return embeddings

def store_in_chromadb(chunks, embeddings):
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"chunk_id": i}]
        )

def search_similar_chunks(question, top_k=3):
    question_embedding = generate_embedding(question)
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    chunks = results['documents'][0]
    scores = results['distances'][0]
    return chunks, scores

def get_answer(question):
    chunks, scores = search_similar_chunks(question)
    context = "\n\n".join(chunks)
    prompt = f"""You are a helpful assistant.
Answer the question using the context below.
Use your best understanding from context.
Give direct specific answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content'], chunks, scores

# ── UI Layout ──────────────────────────

# Title
st.title("🤖 Local RAG System")
st.caption("🔒 Privacy-First — Your documents never leave your system!")
st.divider()

# Two columns
col1, col2 = st.columns([1, 2])

# ── Left Column — Upload & Stats ───────
with col1:
    st.subheader("📄 Upload Document")

    # ── 3 Tabs now! ──────────────────
    tab1, tab2, tab3 = st.tabs([
        "📄 PDF",
        "🌐 Website URL",
        "🗄️ MySQL"
    ])

    # ── PDF Tab ──────────────────────
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload your PDF",
            type=['pdf']
        )
        if uploaded_file is not None:
            file_path = f"./uploaded_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if st.button("⚙️ Process Document", type="primary"):
                if collection.count() > 0:
                    st.info(f"Already processed! ({collection.count()} chunks)")
                else:
                    with st.spinner("Processing document..."):
                        text = load_pdf(file_path)
                        st.write(f"✅ Text: {len(text)} characters")
                        chunks = split_into_chunks(text)
                        st.write(f"✅ Chunks: {len(chunks)}")
                        embeddings = generate_embeddings(chunks)
                        store_in_chromadb(chunks, embeddings)
                    st.success("Document processed! 🎉")

    # ── URL Tab ───────────────────────
    with tab2:
        url_input = st.text_input(
            "Enter Website URL",
            placeholder="https://example.com"
        )
        if st.button("⚙️ Process URL", type="primary"):
            if url_input == "":
                st.error("Please enter a URL first!")
            elif collection.count() > 0:
                st.info(f"Already processed! ({collection.count()} chunks)")
            else:
                with st.spinner("Loading website..."):
                    text = load_from_url(url_input)
                    if text == "":
                        st.error("Failed to load URL!")
                    else:
                        st.write(f"✅ Characters: {len(text)}")
                        chunks = split_into_chunks(text)
                        st.write(f"✅ Chunks: {len(chunks)}")
                        embeddings = generate_embeddings(chunks)
                        store_in_chromadb(chunks, embeddings)
                        st.success("URL Processed! ✅")

    # ── MySQL Tab ─────────────────────
    with tab3:
        st.write("🗄️ Connect MySQL Database")
        
        # Input fields
        host = st.text_input(
            "Host",
            value="localhost"
        )
        user = st.text_input(
            "Username",
            value="root"
        )
        password = st.text_input(
            "Password",
            type="password"  # hides password!
        )
        database = st.text_input(
            "Database Name",
            placeholder="your_database_name"
        )
        
        if st.button("⚙️ Connect & Process", type="primary"):
            if database == "":
                st.error("Please enter database name!")
            elif collection.count() > 0:
                st.info(f"Already processed! ({collection.count()} chunks)")
            else:
                with st.spinner("Connecting to MySQL..."):
                    text = load_from_mysql(
                        host, user, password, database
                    )
                    
                    if "Error" in text:
                        st.error(f"Connection failed! {text}")
                    else:
                        st.write(f"✅ Characters: {len(text)}")
                        chunks = split_into_chunks(text)
                        st.write(f"✅ Chunks: {len(chunks)}")
                        embeddings = generate_embeddings(chunks)
                        store_in_chromadb(chunks, embeddings)
                        st.success("MySQL Processed! ✅")

    st.divider()

    # ── Stats ─────────────────────────
    st.subheader("📊 System Stats")
    st.metric("Total Chunks", collection.count())

    if collection.count() > 0:
        st.success("✅ System Ready!")
    else:
        st.warning("⚠️ Upload a document first!")

    st.divider()

    # ── Clear Button ──────────────────
    if st.button("🗑️ Clear Database"):
        client.delete_collection("pdf_chunks")
        st.session_state.collection = client.get_or_create_collection(
            name="pdf_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        collection = st.session_state.collection
        st.session_state.messages = []
        st.success("Cleared! ✅")
        st.rerun()

        
# ── Right Column — Chat ─────────────────
with col2:
    st.subheader("💬 Ask Questions")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    if question := st.chat_input("Ask anything about your document..."):

        if st.session_state.collection.count() == 0:
            st.error("Please upload and process a document first!")
        else:
            # Show user message
            with st.chat_message("user"):
                st.write(question)
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            # Get answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, chunks, scores = get_answer(question)
                    st.write(answer)

                    # Show sources
                    with st.expander("📚 View Retrieved Chunks"):
                        for i, (chunk, score) in enumerate(zip(chunks, scores)):
                            st.write(f"**Chunk {i+1}** (Score: {round(score, 4)})")
                            st.caption(chunk)
                            st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })