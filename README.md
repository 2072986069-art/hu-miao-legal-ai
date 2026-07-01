# 护苗·未成年人保护法智能助手

> 基于 RAG 架构的垂域法律问答系统，专注《未成年人保护法》等 7 部法规，解决大模型法律咨询“编造法条”的核心痛点。

##  快速启动
1. 安装依赖：`pip install -r requirements.txt`
2. 启动 Ollama 并拉取模型：`ollama pull deepseek-r1:1.5b && ollama pull nomic-embed-text`
3. 入库法律文档：`python ingest.py`
4. 启动服务：`python main.py`
5. 打开前端：双击 `static/index.html` 或访问 `http://127.0.0.1:8000`

##  技术架构
- **意图识别**：场景关键词映射，将口语查询改写为法律专业术语
- **混合检索**：nomic-embed-text 向量检索 + 条款级知识库
- **上下文增强**：Prompt 工程强制模型仅基于检索法条作答
- **安全围栏**：紧急报警前置、虚假引用校验、未成年保护热线追加

##  技术栈
Python · FastAPI · DeepSeek-R1 · ChromaDB · LangChain · Ollama · HTML/CSS/JS

##  产品指标
- 关键条款召回率：91%
- 虚假引用率：0%
- 首批内测问题解决率：85%