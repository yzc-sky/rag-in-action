import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI as OpenAIClient  # 避免与TruLens的OpenAI类名冲突
from trulens.core import TruSession, Feedback, Select
from trulens.apps.app import TruApp, instrument
from trulens.providers.openai import OpenAI as TruLensOpenAI
import numpy as np

# 设置API密钥
# os.environ["OPENAI_API_KEY"] = "your_key_here"

# 初始化嵌入函数
embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get("OPENAI_API_KEY"),
                                             model_name="text-embedding-ada-002")
chroma_client = chromadb.Client()
vector_store = chroma_client.get_or_create_collection("Info", embedding_function=embedding_function)

# 添加示例数据
vector_store.add("starbucks_info", documents=[
    """
    Starbucks Corporation is an American multinational chain of coffeehouses headquartered in Seattle, Washington.
    As the world's largest coffeehouse chain, Starbucks is seen to be the main representation of the United States' second wave of coffee culture.
    """
])

class RAG:
    @instrument  # TruLens装饰器，用于跟踪函数调用
    def retrieve(self, query: str):
        """检索相关文档"""
        results = vector_store.query(query_texts=[query], n_results=2)
        return results["documents"][0] if results["documents"] else []

    @instrument
    def generate_completion(self, query: str, context: list):
        """生成回答"""
        oai_client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY"))
        context_str = "\n".join(context) if context else "No context available."
        completion = oai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Context: {context_str}\nQuestion: {query}"}]
        ).choices[0].message.content
        return completion

    @instrument
    def query(self, query: str):
        """完整的RAG查询流程"""
        context = self.retrieve(query)
        return self.generate_completion(query, context)

# 初始化TruLens会话
session = TruSession(database_redact_keys=True)
session.reset_database()

# 初始化TruLens的OpenAI提供者
provider = TruLensOpenAI(model_engine="gpt-4")

# 定义评估指标
f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness") \
    .on(Select.RecordCalls.retrieve.rets).on_output()
f_answer_relevance = Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance") \
    .on_input().on_output()
f_context_relevance = Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance") \
    .on_input().on(Select.RecordCalls.retrieve.rets[:]).aggregate(np.mean)

# 设置TruApp
rag = RAG()
tru_rag = TruApp(
    rag,
    app_name="RAG",
    app_version="base",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance]
)

# 执行查询并记录
with tru_rag as recording:
    response = rag.query("What wave of coffee culture is Starbucks seen to represent in the United States?")
    print(f"Response: {response}")

# 查看评估结果
print(session.get_leaderboard())
