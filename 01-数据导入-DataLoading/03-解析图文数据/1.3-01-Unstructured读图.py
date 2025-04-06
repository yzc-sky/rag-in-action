from langchain_community.document_loaders import UnstructuredImageLoader
image_path = "data/黑神话/黑悟空英文.jpg"
loader = UnstructuredImageLoader(image_path)

data = loader.load()
print(data)
