from langchain_deepseek import ChatDeepSeek 
from langchain_huggingface import HuggingFaceEmbeddings 
# 初始化语言模型和向量嵌入模型
llm = ChatDeepSeek(model="deepseek-chat", temperature=0.1)
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
# 准备游戏知识文本，创建Document对象。
from langchain.schema import Document
game_knowledge = """
《灭神纪∙猢狲》是一款动作角色扮演游戏……
"""
# 创建Document对象
documents = [Document(page_content=game_knowledge)]
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 父文本块分割器（较大的文本块）
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", "！", "？", "；", ",", " ", ""]
)
# 子文本块分割器（较小的文本块）
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "；", ",", " ", ""]
)
# 创建父子文本块
parent_docs = parent_splitter.split_documents(documents)
child_docs = child_splitter.split_documents(documents)
# 创建存储和检索器，建立两层存储系统
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
vectorstore = Chroma(
    collection_name="game_knowledge",
    embedding_function=embed_model
)
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
# 添加文本块
retriever.add_documents(documents)
# 自定义提示模板
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
prompt_template = """基于以下上下文信息回答问题。如果无法找到答案，请说“我找不到相关信息”。
上下文：
{context}
问题：{question}
回答："""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)
# 创建问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # 问答链类型
    retriever=retriever,# 检索器
    return_source_documents=True, # 是否返回源文档
    chain_type_kwargs={"prompt": PROMPT}
)
# 通过实际问答测试系统
test_questions = [
    "游戏中悟空有哪些形态变化？",
    "游戏的画面风格是怎样的？",
]
for question in test_questions:
    print(f"\n问题：{question}")
    result = qa_chain({"query": question})    
    print(f"\n回答：{result['result']}")
    print("\n使用的源文档：")
    for i, doc in enumerate(result["source_documents"], 1):
        print(f"\n相关文档 {i}:")
        print(f"长度：{len(doc.page_content)} 字符")
        print(f"内容片段：{doc.page_content[:150]}...")
        print("---")
