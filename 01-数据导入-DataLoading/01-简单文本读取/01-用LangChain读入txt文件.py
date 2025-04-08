# 读取单个txt文件
from langchain_community.document_loaders import TextLoader
loader = TextLoader("90-文档-Data/黑悟空/设定.txt")
documents = loader.load()
print(documents)
