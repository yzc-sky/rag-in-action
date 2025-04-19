from pymilvus import MilvusClient, DataType
import random
import time

# 1. 连接到Milvus服务器
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

# 2. 创建集合（如果不存在）
collection_name = "ann_search_demo"
dim = 128

# 检查集合是否存在
if client.has_collection(collection_name):
    client.drop_collection(collection_name)

# 创建 schema
schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)

# 创建集合
client.create_collection(
    collection_name=collection_name,
    schema=schema
)

# 3. 准备测试数据
num_vectors = 1000
vectors = [[random.random() for _ in range(dim)] for _ in range(num_vectors)]
ids = [i for i in range(num_vectors)]
entities = [{"id": ids[i], "vector": vectors[i]} for i in range(num_vectors)]

# 插入数据
client.insert(
    collection_name=collection_name,
    data=entities
)

# 等待数据写入完成
time.sleep(2)

# 创建索引
client.create_index(
    collection_name=collection_name,
    field_name="vector",
    index_type="IVF_FLAT",
    metric_type="L2",
    params={"nlist": 128}  # 减小nlist值以适应较小的数据集
)

# 等待索引创建完成
time.sleep(2)

# 加载集合
client.load_collection(collection_name)

# 4. 单向量搜索示例
def single_vector_search():
    print("\n=== 单向量搜索示例 ===")
    query_vector = [random.random() for _ in range(dim)]
    
    res = client.search(
        collection_name=collection_name,
        data=[query_vector],
        anns_field="vector",
        limit=3,
        output_fields=["id"],
        search_params={
            "metric_type": "L2",
            "params": {"nprobe": 10}  # 添加搜索参数
        }
    )
    
    print("搜索结果:")
    for hits in res:
        for hit in hits:
            print(f"ID: {hit['id']}, 距离: {hit['distance']}")

# 5. 批量向量搜索示例
def batch_vector_search():
    print("\n=== 批量向量搜索示例 ===")
    query_vectors = [[random.random() for _ in range(dim)] for _ in range(2)]
    
    res = client.search(
        collection_name=collection_name,
        data=query_vectors,
        anns_field="vector",
        limit=3,
        output_fields=["id"],
        search_params={
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
    )
    
    print("批量搜索结果:")
    for i, hits in enumerate(res):
        print(f"\n查询向量 {i+1} 的结果:")
        for hit in hits:
            print(f"ID: {hit['id']}, 距离: {hit['distance']}")

# 6. 带过滤条件的搜索示例
def filtered_search():
    print("\n=== 带过滤条件的搜索示例 ===")
    query_vector = [random.random() for _ in range(dim)]
    
    res = client.search(
        collection_name=collection_name,
        data=[query_vector],
        anns_field="vector",
        limit=3,
        output_fields=["id"],
        expr="id > 500",  # 只搜索ID大于500的向量
        search_params={
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
    )
    
    print("过滤搜索结果:")
    for hits in res:
        for hit in hits:
            print(f"ID: {hit['id']}, 距离: {hit['distance']}")

# 7. 分页搜索示例
def paginated_search():
    print("\n=== 分页搜索示例 ===")
    query_vector = [random.random() for _ in range(dim)]
    page_size = 3
    page_num = 2
    
    res = client.search(
        collection_name=collection_name,
        data=[query_vector],
        anns_field="vector",
        limit=page_size,
        offset=(page_num-1)*page_size,
        output_fields=["id"],
        search_params={
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
    )
    
    print(f"第 {page_num} 页搜索结果:")
    for hits in res:
        for hit in hits:
            print(f"ID: {hit['id']}, 距离: {hit['distance']}")

# 运行所有示例
if __name__ == "__main__":
    try:
        single_vector_search()
        batch_vector_search()
        filtered_search()
        paginated_search()
    finally:
        # 清理
        client.release_collection(collection_name)
        client.drop_collection(collection_name)
