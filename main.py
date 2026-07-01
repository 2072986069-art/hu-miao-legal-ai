# main.py
import os
import re
import ollama
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

# ==================== 配置 ====================
BASE_DIR = r"D:\hu_miao_env"
CHROMA_PATH = r"D:\hu_miao_db\chroma"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8020

app = FastAPI(title="护苗法律助手", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 向量库（使用 Ollama 嵌入） ====================
embedding_func = embedding_functions.OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434/api/embeddings"
)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name="child_protection_laws",
    embedding_function=embedding_func
)

# ==================== 检索函数（纯向量检索，不依赖 where_document） ====================
def retrieve(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """根据查询返回最相关的文档片段"""
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    docs = []
    if results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            docs.append({
                "text": results['documents'][0][i],
                "source": results['metadatas'][0][i].get("source", "未知"),
                "relevance": 1 - results['distances'][0][i]  # 余弦距离转相似度
            })
    # 按相似度从高到低排序
    docs.sort(key=lambda x: x['relevance'], reverse=True)
    return docs[:top_k]

# ==================== 意图识别与查询改写 ====================
def rewrite_query(original_query: str) -> str:
    """根据场景关键词追加专业术语，提升检索精准度"""
    scene_map = {
        "家长打": "家庭暴力 监护侵害 强制报告",
        "爸爸打": "家庭暴力 监护侵害 强制报告",
        "妈妈打": "家庭暴力 监护侵害 强制报告",
        "家暴": "家庭暴力 强制报告 监护侵害",
        "被家长": "家庭暴力 强制报告 监护侵害",
        "被父母": "家庭暴力 强制报告 监护侵害",
        "同学打": "校园欺凌 学生欺凌 学校责任",
        "欺凌": "校园欺凌 学生欺凌防控 学校责任",
        "校园暴力": "校园欺凌 学校责任 第三十九条",
        "强制报告": "侵害未成年人案件强制报告制度 报告义务",
        "网络": "未成年人网络保护 网络沉迷 游戏充值",
        "游戏充值": "未成年人网络消费 退款 民事行为能力",
        "被开除": "未成年人受教育权 学校开除 变相开除",
    }
    query_lower = original_query.lower()
    for keyword, addition in scene_map.items():
        if keyword in query_lower:
            return f"{original_query} {addition}"
    return original_query

# ==================== System Prompt ====================
SYSTEM_PROMPT = """你是“护苗”法律助手，只能依据下面提供的【参考资料】回答用户问题。

规则：
1. 如果用户问“怎么办”“如何处理”，请按【时间顺序】给出具体操作步骤（如：立即制止→通知家长→心理辅导→报告上级等）。
2. 每个步骤都要引用具体法条，格式：[法律名称第X条]。
3. 回答简洁，每步一句话，不要泛泛而谈。
4. 如果涉及人身危险，第一步必须是“立即拨打110报警”。
5. 如果资料中无相关条款，明确说明“现有资料未覆盖，建议咨询专业律师”。
6. 语气专业温和，对未成年人使用保护性口吻。
7. 当问题涉及家长对子女的暴力行为时，必须优先引用《家庭教育促进法》第二十三条（禁止家庭暴力）和《关于建立侵害未成年人案件强制报告制度的意见》相关条款。"""

# ==================== 构建提示词 ====================
def build_prompt(query: str, docs: List[Dict[str, Any]], history: List[Dict[str, str]] = []) -> str:
    ctx = "\n\n---\n".join([f"[来源：{d['source']}]\n{d['text']}" for d in docs])
    hist = ""
    if history:
        hist = "【对话历史】\n" + "\n".join(
            [f"用户: {h['user']}\n助手: {h['assistant']}" for h in history[-3:]]
        ) + "\n\n"
    return f"{SYSTEM_PROMPT}\n\n{hist}【参考资料】\n{ctx}\n\n【用户问题】\n{query}\n请回答："

# ==================== 调用本地模型生成回答 ====================
def generate_response(prompt: str) -> str:
    try:
        resp = ollama.chat(
            model='deepseek-r1:1.5b',
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 512}
        )
        answer = resp['message']['content']
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        return answer if answer else "（生成空内容，请检查）"
    except Exception as e:
        return f"模型调用失败: {str(e)}"

# ==================== 安全过滤 ====================
def safety_filter(answer: str, is_minor: bool = False) -> str:
    for kw in ["规避法律", "销毁证据", "私下报复", "如何不被发现", "逃脱责任"]:
        if kw in answer:
            return "抱歉，该问题涉及不当内容，无法提供回答。"
    if is_minor:
        answer += "\n\n💡 如果你正在经历困难或危险，请告诉家长、老师，或拨打12345未成年人保护热线。"
    return answer

# ==================== 接口模型 ====================
class ChatRequest(BaseModel):
    message: str
    is_minor: bool = False
    history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    answer: str
    references: List[Dict[str, Any]]

# ==================== 聊天接口 ====================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # 1. 意图识别与查询改写（用于检索）
    rewritten = rewrite_query(req.message)
    # 2. 检索相关文档
    docs = retrieve(rewritten, top_k=6)
    # 3. 构建最终提示词（保留原始问题，让模型理解真实语境）
    prompt = build_prompt(req.message, docs, req.history)
    # 4. 生成回答
    raw = generate_response(prompt)
    # 5. 安全后处理
    final = safety_filter(raw, req.is_minor)
    # 6. 整理引用信息
    refs = [{"source": d["source"], "snippet": d["text"][:150] + "..."} for d in docs]
    return ChatResponse(answer=final, references=refs)

@app.get("/health")
async def health():
    return {"status": "ok", "model": "deepseek-r1:1.5b", "db_docs": collection.count()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)