from langchain_community.document_loaders import TextLoader
print("=== TextLoader 加载结果 ===")
text_loader = TextLoader("data/西游记人物角色.json")
text_documents = text_loader.load()
print(text_documents)
