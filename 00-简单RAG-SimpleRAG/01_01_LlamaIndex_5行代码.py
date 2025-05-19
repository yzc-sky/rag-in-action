"""
注意：运行此代码前，请确保已在环境变量中设置OpenAI API密钥。
在Linux/Mac系统中，可以通过以下命令设置：
export OPENAI_API_KEY='your-api-key'

在Windows系统中，可以通过以下命令设置：
set OPENAI_API_KEY=your-api-key

如果无法取得OpenAI API密钥，也没关系，我们有平替方案，请移步至其它程序。
"""

# 第一行代码：导入相关的库
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
import os
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# 配置嵌入模型
embed_model = OpenAIEmbedding(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

# 配置LLM模型
llm = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

# 设置全局配置
Settings.embed_model = embed_model
Settings.llm = llm

# 第二行代码：加载数据
documents = SimpleDirectoryReader(input_files=["90-文档-Data/黑悟空/设定.txt"]).load_data() 
# 第三行代码：构建索引
index = VectorStoreIndex.from_documents(documents)
# 第四行代码：创建问答引擎
query_engine = index.as_query_engine()
# 第五行代码: 开始问答
print(query_engine.query("黑神话悟空中有哪些战斗工具?"))
