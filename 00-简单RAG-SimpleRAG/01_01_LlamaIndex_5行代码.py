# 第一行代码：导入相关的库
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader 
# 第二行代码：加载数据
documents = SimpleDirectoryReader(input_files=["90-文档-Data/黑悟空/设定.txt"]).load_data() 
# 第三行代码：构建索引
index = VectorStoreIndex.from_documents(documents)
# 第四行代码：创建问答引擎
query_engine = index.as_query_engine()
# 第五行代码: 开始问答
print(query_engine.query("黑神话悟空中有哪些战斗工具?"))
