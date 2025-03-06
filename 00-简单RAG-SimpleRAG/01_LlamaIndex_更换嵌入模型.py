# 导入相关的库
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding # 需要pip install llama-index-embeddings-huggingface

# 加载本地嵌入模型
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh" # 模型路径和名称（首次执行时会从HuggingFace下载）
    )

# 加载数据
documents = SimpleDirectoryReader("data/黑悟空设定").load_data() 

# 构建索引
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

# 创建问答引擎
query_engine = index.as_query_engine()

# 开始问答
print(query_engine.query("黑神话悟空中有哪些战斗工具?"))