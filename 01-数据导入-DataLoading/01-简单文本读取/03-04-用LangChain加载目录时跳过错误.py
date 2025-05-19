from langchain_community.document_loaders import DirectoryLoader, TextLoader
# 加载目录下所有文件，跳过出错文件，因为有些文件是图片，TextLoader 无法加载
loader = DirectoryLoader("./90-文档-Data/黑悟空",
                         silent_errors=True,
                         loader_cls=TextLoader
                         )

docs = loader.load()
print(docs[0].page_content[:100])  # 打印第一个文档内容的前100个字符
