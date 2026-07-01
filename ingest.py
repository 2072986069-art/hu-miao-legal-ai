# ingest.py
import os
import re
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = r"D:\hu_miao_env"
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_PATH = r"D:\hu_miao_db\chroma"
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            parts.append(txt)
    return "\n".join(parts)

def split_by_articles(text):
    pattern = r'(第[一二三四五六七八九十百千\d]+条[\s\S]*?)(?=第[一二三四五六七八九十百千\d]+条|$)'
    articles = re.findall(pattern, text)
    return articles if articles else [text]

def chunk_text(text, max_len=400, overlap=50):
    if len(text) <= max_len:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        chunks.append(text[start:end])
        start += (max_len - overlap)
    return chunks

print("📚 处理法律文档...")
all_chunks = []
source_files = [
    "未成年人保护法.pdf",
    "学校保护规定.pdf",
    "预防犯罪法.pdf",
    "典型案例.txt"
]

for src in source_files:
    path = os.path.join(DOCS_DIR, src)
    if not os.path.exists(path):
        print(f"⚠️ 跳过: {path}")
        continue
    print(f"📄 {src}")
    if src.endswith('.pdf'):
        text = extract_text_from_pdf(path)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
    articles = split_by_articles(text)
    for art in articles:
        for sub in chunk_text(art):
            if len(sub.strip()) > 10:
                all_chunks.append({"text": sub, "source": src})

print(f"✅ 共 {len(all_chunks)} 个片段，开始向量化...")

# 使用 Ollama 嵌入，无需联网
embedding_func = embedding_functions.OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434/api/embeddings"
)

client = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    collection = client.get_collection("child_protection_laws", embedding_function=embedding_func)
    ids = collection.get()['ids']
    if ids:
        collection.delete(ids=ids)
        print(f"🧹 已清空旧向量 {len(ids)} 条")
except Exception:
    collection = client.create_collection("child_protection_laws", embedding_function=embedding_func)

batch_size = 50
for i in range(0, len(all_chunks), batch_size):
    batch = all_chunks[i:i+batch_size]
    collection.add(
        documents=[b["text"] for b in batch],
        metadatas=[{"source": b["source"]} for b in batch],
        ids=[f"doc_{i+j}" for j in range(len(batch))]
    )
    print(f"进度: {min(i+batch_size, len(all_chunks))}/{len(all_chunks)}")

print(f"🎉 入库完成！向量总数: {collection.count()}")