from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

# 定义不同业务场景的提示模板
templates = {
    "customer_service": """
    你是一个专业的客服代表。请根据以下客户反馈和历史案例提供解决方案：
    
    历史案例：
    {similar_cases}
    
    当前客户反馈：{customer_feedback}
    
    请提供：
    1. 问题分析
    2. 具体解决方案
    3. 预防措施
    """,
    
    "technical_support": """
    你是一个技术专家。请根据以下技术问题和历史案例提供解决方案：
    
    历史案例：
    {similar_cases}
    
    当前问题描述：{issue_description}
    
    请提供：
    1. 问题诊断
    2. 解决步骤
    3. 技术建议
    """,
    
    "business_analysis": """
    你是一个商业分析师。请根据以下业务问题和历史案例提供分析：
    
    历史案例：
    {similar_cases}
    
    当前问题描述：{business_issue}
    
    请提供：
    1. 问题分析
    2. 影响评估
    3. 改进建议
    """
}

# 示例数据库
case_database = {
    "customer_service": [
        "客户反馈产品包装破损，我们提供了免费换货服务，并改进了包装材料",
        "客户投诉配送延迟，我们优化了物流路线，并提供了补偿方案",
        "客户反映产品质量问题，我们进行了质量检查，并提供了退款服务"
    ],
    "technical_support": [
        "系统频繁崩溃，通过更新服务器配置和优化代码解决了问题",
        "数据库连接超时，通过增加连接池和优化查询语句解决了问题",
        "API响应慢，通过添加缓存和优化算法提高了性能"
    ],
    "business_analysis": [
        "销售额下降，通过市场调研发现是竞争对手价格战导致，调整了营销策略",
        "客户流失率高，通过客户满意度调查发现是服务体验问题，改进了服务流程",
        "运营成本上升，通过流程优化和自动化降低了成本"
    ]
}

# 创建路由函数
def get_prompt_template(scenario):
    if scenario in templates:
        return PromptTemplate.from_template(templates[scenario])
    else:
        raise ValueError(f"未知的场景: {scenario}")

# 获取相似案例
def get_similar_cases(scenario, query, k=2):
    # 创建向量数据库
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    db = FAISS.from_texts(case_database[scenario], embeddings)
    
    # 检索相似案例
    docs = db.similarity_search(query, k=k)
    return "\n".join([f"- {doc.page_content}" for doc in docs])

# 示例使用

# 初始化LLM
llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

# 测试不同场景的路由选择
scenarios = ["customer_service", "technical_support", "business_analysis"]
test_queries = {
    "customer_service": "产品包装破损，导致商品损坏",
    "technical_support": "系统频繁崩溃，错误代码500", 
    "business_analysis": "销售额连续三个月下降"
}

# 遍历测试每个场景
for scenario in scenarios:
    query = test_queries[scenario]
    print(f"\n{'='*20} {scenario} {'='*20}")
    print(f"输入问题: {query}")
    
    # 获取对应的提示模板
    prompt_template = get_prompt_template(scenario)
    print("\n选择的提示模板:")
    print(prompt_template.template)
    
    # 获取相似案例
    similar_cases = get_similar_cases(scenario, query)
    print("\n检索到的相似案例:")
    print(similar_cases)
    
    # 根据场景使用不同的参数名
    params = {
        "customer_service": {"customer_feedback": query},
        "technical_support": {"issue_description": query},
        "business_analysis": {"business_issue": query}
    }
    
    # 生成回复
    params[scenario]["similar_cases"] = similar_cases
    response = llm.invoke(prompt_template.format(**params[scenario]))
    print("\n生成的回复:")
    print(response)


