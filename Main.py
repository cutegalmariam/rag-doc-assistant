import os
from dotenv import load_dotenv
from openai import OpenAI
import fitz
import faiss
import numpy as np

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# globals for FAISS
INDEX = None
CHUNKS = []


def extract_text_from_file(file_path):
    if file_path.endswith(".txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        return "\n".join([page.get_text() for page in doc])
    else:
        raise ValueError("Unsupported file type: only .txt or .pdf allowed.")

# chunk the text into smaller parts
def chunk_text(text, max_tokens=500):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_tokens:
            current_chunk += sentence + '. '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# embed and index the text chunks using FAISS
def create_vector_index(chunks):
    global INDEX, CHUNKS
    embeddings = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )
    vectors = np.array([e.embedding for e in embeddings.data]).astype('float32')
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(vectors)
    INDEX = index
    CHUNKS = chunks

# find top-k most relevant chunks
def retrieve_relevant_chunks(query, k=3):
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    ).data[0].embedding
    embedding = np.array([embedding]).astype('float32')
    distances, indices = INDEX.search(embedding, k)
    return [CHUNKS[i] for i in indices[0]]


def generate_answer_with_context(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = (
        f"You are a helpful assistant. Use the following document context to answer the user's question.\n\n"
        f"Context:\n{context}\n\n"
        f"If the answer is not found in the context, say: 'I couldn't find that in the documents.'\n\n"
        f"Question: {query}"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

def main():
    print("📄 Load a .txt or .pdf file to start.")
    file_path = input("Enter the file path: ").strip()
    try:
        full_text = extract_text_from_file(file_path)
        print("✅ File loaded. Processing...")
        chunks = chunk_text(full_text)
        create_vector_index(chunks)
        print(f"📚 Document indexed with {len(chunks)} chunks. You can now ask questions!\n")
    except Exception as e:
        print("❌ Error loading file:", e)
        return

    while True:
        query = input("🧠 You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        context_chunks = retrieve_relevant_chunks(query)
        answer = generate_answer_with_context(query, context_chunks)
        print("🤖 GPT:", answer, "\n")

if __name__ == "__main__":
    main()
