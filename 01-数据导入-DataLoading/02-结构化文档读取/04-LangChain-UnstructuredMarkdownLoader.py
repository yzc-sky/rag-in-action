from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

markdown_path = "data/黑神话/黑悟空版本介绍.md"
loader = UnstructuredMarkdownLoader(markdown_path)

data = loader.load()
print(data[0].page_content[:250])

loader = UnstructuredMarkdownLoader(markdown_path, mode="elements")
data = loader.load()
print(f"Number of documents: {len(data)}\n")
for document in data:
    print(f"{document}\n")
