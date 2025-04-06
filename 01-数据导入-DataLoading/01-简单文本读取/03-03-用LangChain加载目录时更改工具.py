from langchain_community.document_loaders import DirectoryLoader, TextLoader
# 加载目录下所有 Markdown 文件
loader = DirectoryLoader("data/黑神话",
                         glob="**/*.md",
                         loader_cls=TextLoader # 指定加载工具
                         )
docs = loader.load()
print(docs[0].page_content[:100])  # 打印第一个文档内容的前100个字符
