# 导入相关的库
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain.load import dumps, loads
# 加载文档
doc_dir = "90-文档-Data/山西文旅"
def load_documents(directory):
    """读取目录中的所有文档（包括PDF、TXT、DOCX)"""
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
        else:
            continue  # 跳过不支持的文件类型
        documents.extend(loader.load())
    return documents
docs = load_documents(doc_dir)
# 文本切块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
splits = text_splitter.split_documents(docs)
# 获取嵌入并创建向量索引
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)
retriever = vectorstore.as_retriever()
# RRF算法
def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results
# 生成多个搜索查询
template = """你是一个帮助用户生成多个搜索查询的助手。\n
请根据以下问题生成多个相关的搜索查询：{question} \n
输出（4个查询）："""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)
llm = ChatDeepSeek(model="deepseek-chat")
generate_queries = (
    prompt_rag_fusion 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)
# 示例问题
questions = [
    "山西有哪些著名的旅游景点？",
    "云冈石窟的历史背景是什么？",
    "五台山的文化和宗教意义是什么？"
]
# 进行检索和RRF处理
for question in questions:
    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
    docs = retrieval_chain_rag_fusion.invoke({"question": question})
    
    print(f"\n【问题】{question}")
    print(f"文档数量：{len(docs)}")
    for doc, score in docs[:3]:  # 显示前3个结果
        print(f"文档内容：{doc.page_content[:200]}...")  # 只展示前200个字符
