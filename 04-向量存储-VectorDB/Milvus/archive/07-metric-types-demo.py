from pymilvus import MilvusClient, utility
import random

# 1. 连接到Milvus服务器
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

# 2. 演示L2距离度量
print("\n=== L2距离度量演示 ===")
collection_name = "l2_demo"
dim = 128

# 创建集合
if not utility.has_collection(collection_name):
    client.create_collection(
        collection_name=collection_name,
        dimension=dim,
        metric_type="L2"
    )

# 生成测试数据
vectors = [[random.random() for _ in range(dim)] for _ in range(1000)]
ids = [i for i in range(1000)]

# 插入数据
client.insert(collection_name=collection_name, data=vectors, ids=ids)

# 搜索
query_vector = [random.random() for _ in range(dim)]
res = client.search(
    collection_name=collection_name,
    data=[query_vector],
    limit=3,
    search_params={"metric_type": "L2"}
)

print("L2距离搜索结果:")
for hits in res:
    for hit in hits:
        print(f"ID: {hit['id']}, 距离: {hit['distance']}")

# 清理
client.drop_collection(collection_name)

# 3. 演示内积度量
print("\n=== 内积度量演示 ===")
collection_name = "ip_demo"

# 创建集合
if not utility.has_collection(collection_name):
    client.create_collection(
        collection_name=collection_name,
        dimension=dim,
        metric_type="IP"
    )

# 插入数据
client.insert(collection_name=collection_name, data=vectors, ids=ids)

# 搜索
res = client.search(
    collection_name=collection_name,
    data=[query_vector],
    limit=3,
    search_params={"metric_type": "IP"}
)

print("内积搜索结果:")
for hits in res:
    for hit in hits:
        print(f"ID: {hit['id']}, 距离: {hit['distance']}")

# 清理
client.drop_collection(collection_name)

# 4. 演示余弦相似度度量
print("\n=== 余弦相似度度量演示 ===")
collection_name = "cosine_demo"

# 创建集合
if not utility.has_collection(collection_name):
    client.create_collection(
        collection_name=collection_name,
        dimension=dim,
        metric_type="COSINE"
    )

# 插入数据
client.insert(collection_name=collection_name, data=vectors, ids=ids)

# 搜索
res = client.search(
    collection_name=collection_name,
    data=[query_vector],
    limit=3,
    search_params={"metric_type": "COSINE"}
)

print("余弦相似度搜索结果:")
for hits in res:
    for hit in hits:
        print(f"ID: {hit['id']}, 距离: {hit['distance']}")

# 清理
client.drop_collection(collection_name)

# 5. 演示Jaccard距离度量
print("\n=== Jaccard距离度量演示 ===")
collection_name = "jaccard_demo"
dim = 64  # 二进制向量的维度必须是8的倍数

# 创建集合
if not utility.has_collection(collection_name):
    client.create_collection(
        collection_name=collection_name,
        dimension=dim,
        metric_type="JACCARD",
        vector_type="BINARY_VECTOR"
    )

# 生成二进制测试数据
binary_vectors = [[random.randint(0, 1) for _ in range(dim)] for _ in range(1000)]
binary_ids = [i for i in range(1000)]

# 插入数据
client.insert(collection_name=collection_name, data=binary_vectors, ids=binary_ids)

# 搜索
query_binary = [random.randint(0, 1) for _ in range(dim)]
res = client.search(
    collection_name=collection_name,
    data=[query_binary],
    limit=3,
    search_params={"metric_type": "JACCARD"}
)

print("Jaccard距离搜索结果:")
for hits in res:
    for hit in hits:
        print(f"ID: {hit['id']}, 距离: {hit['distance']}")

# 清理
client.drop_collection(collection_name)

# 6. 演示汉明距离度量
print("\n=== 汉明距离度量演示 ===")
collection_name = "hamming_demo"

# 创建集合
if not utility.has_collection(collection_name):
    client.create_collection(
        collection_name=collection_name,
        dimension=dim,
        metric_type="HAMMING",
        vector_type="BINARY_VECTOR"
    )

# 插入数据
client.insert(collection_name=collection_name, data=binary_vectors, ids=binary_ids)

# 搜索
res = client.search(
    collection_name=collection_name,
    data=[query_binary],
    limit=3,
    search_params={"metric_type": "HAMMING"}
)

print("汉明距离搜索结果:")
for hits in res:
    for hit in hits:
        print(f"ID: {hit['id']}, 距离: {hit['distance']}")

# 清理
client.drop_collection(collection_name) 