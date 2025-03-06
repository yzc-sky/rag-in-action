# 导入相关的库
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # 需要pip install llama-index-embeddings-huggingface
from llama_index.llms.deepseek import DeepSeek  # 需要pip install llama-index-llms-deepseek

# 加载本地嵌入模型
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh")

# 加载环境变量
from dotenv import load_dotenv
import os

# 加载 .env 文件中的环境变量
load_dotenv()

# 创建 Deepseek LLM（通过API调用最新的DeepSeek大模型）
llm = DeepSeek(
    model="deepseek-chat", # 目前V3，如使用deepseek-reasoner则为最新推理模型R1
    api_key=os.getenv("DEEPSEEK_API_KEY")  # 从环境变量获取API key
)

# 加载数据
documents = SimpleDirectoryReader("data/黑悟空").load_data() 

# 构建索引
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    # llm=llm  # 设置构建索引时的语言模型（一般不需要）
)

# 创建问答引擎
query_engine = index.as_query_engine(
    llm=llm  # 设置生成模型
    )

# 开始问答
print(query_engine.query("黑神话悟空中有哪些战斗工具?"))