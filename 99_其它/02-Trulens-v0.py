import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
from trulens.core import TruSession, Feedback, Select
from trulens.apps.app import TruApp
from trulens.providers.openai import OpenAI as TruOpenAI

# ✅ API KEY 设置
# os.environ["OPENAI_API_KEY"] = "sk-..."  # 替换为你的 API Key

# ✅ 初始化向量库
embedding_fn = OpenAIEmbeddingFunction(api_key=os.environ["OPENAI_API_KEY"])
client = chromadb.Client()
collection = client.get_or_create_collection("demo", embedding_function=embedding_fn)

# ✅ 添加知识文本
collection.add(documents=["Starbucks 是美国第二波咖啡文化的代表。"], ids=["doc1"])

# ✅ 基础 RAG 类
from trulens.apps import instrument
class RAG:
    @instrument
    def retrieve(self, query: str):
        results = collection.query(query_texts=[query], n_results=2)
        return [doc for sub in results["documents"] for doc in sub]

    @instrument
    def generate_completion(self, query: str, context: list):
        ctx = "\n".join(context)
        prompt = f"上下文：\n{ctx}\n\n请回答：{query}"
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    @instrument
    def query(self, query: str):
        ctx = self.retrieve(query)
        return self.generate_completion(query, ctx)

rag = RAG()

# ✅ 初始化 TruLens 会话
session = TruSession()
session.reset_database()
provider = TruOpenAI(model_engine="gpt-4")

# ✅ 定义反馈函数
f_grounded = Feedback(provider.groundedness_measure_with_cot_reasons, name="Grounded") \
    .on(Select.RecordCalls.retrieve.rets.collect()) \
    .on_output()

# ✅ 包装为 Tru 应用
tru_app = TruApp(
    app=rag,
    app_name="MiniRAG",
    feedbacks=[f_grounded]
)

# ✅ 开始记录
with tru_app as recording:
    print(rag.query("Starbucks 属于哪一波咖啡文化？"))

# ✅ 查看反馈结果
session.get_leaderboard()
