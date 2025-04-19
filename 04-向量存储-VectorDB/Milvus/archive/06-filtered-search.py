from pymilvus import MilvusClient
import random

# 初始化Milvus客户端
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

# 创建集合
collection_name = "filtered_search_demo"
schema = {
    "fields": [
        {"name": "id", "dtype": "INT64", "is_primary": True},
        {"name": "vector", "dtype": "FLOAT_VECTOR", "dim": 5},
        {"name": "color", "dtype": "VARCHAR", "max_length": 100},
        {"name": "likes", "dtype": "INT64"}
    ]
}

# 如果集合已存在，先删除
if client.has_collection(collection_name):
    client.drop_collection(collection_name)

# 创建集合
client.create_collection(collection_name, schema)

# 准备测试数据
entities = [
    {"id": 0, "vector": [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592], "color": "pink_8682", "likes": 165},
    {"id": 1, "vector": [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104], "color": "red_7025", "likes": 25},
    {"id": 2, "vector": [0.43742130801983836, -0.5597502546264526, 0.6457887650909682, 0.7894058910881185, 0.20785793220625592], "color": "orange_6781", "likes": 764},
    {"id": 3, "vector": [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995, 0.95791889146345], "color": "pink_9298", "likes": 234},
    {"id": 4, "vector": [0.4452349528804562, -0.8757026943054742, 0.8220779437047674, 0.46406290649483184, 0.30337481143159106], "color": "red_4794", "likes": 122},
    {"id": 5, "vector": [0.985825131989184, -0.8144651566660419, 0.6299267002202009, 0.1206906911183383, -0.1446277761879955], "color": "yellow_4222", "likes": 12},
    {"id": 6, "vector": [0.8371977790571115, -0.015764369584852833, -0.31062937026679327, -0.562666951622192, -0.8984947637863987], "color": "red_9392", "likes": 58},
    {"id": 7, "vector": [-0.33445148015177995, -0.2567135004164067, 0.8987539745369246, 0.9402995886420709, 0.5378064918413052], "color": "grey_8510", "likes": 775},
    {"id": 8, "vector": [0.39524717779832685, 0.4000257286739164, -0.5890507376891594, -0.8650502298996872, -0.6140360785406336], "color": "white_9381", "likes": 876},
    {"id": 9, "vector": [0.5718280481994695, 0.24070317428066512, -0.3737913482606834, -0.06726932177492717, -0.6980531615588608], "color": "purple_4976", "likes": 765}
]

# 插入数据
client.insert(collection_name, entities)

# 创建索引
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}
client.create_index(collection_name, "vector", index_params)

# 加载集合
client.load_collection(collection_name)

# 1. 标准过滤搜索示例
print("\n=== 标准过滤搜索示例 ===")
query_vector = [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]

res = client.search(
    collection_name=collection_name,
    data=[query_vector],
    limit=5,
    filter='color like "red%" and likes > 50',
    output_fields=["color", "likes"]
)

print("标准过滤搜索结果:")
for hits in res:
    for hit in hits:
        print(f"ID: {hit['id']}, 距离: {hit['distance']:.4f}, 颜色: {hit['entity']['color']}, 点赞数: {hit['entity']['likes']}")

# 2. 迭代过滤搜索示例
print("\n=== 迭代过滤搜索示例 ===")
res = client.search(
    collection_name=collection_name,
    data=[query_vector],
    limit=5,
    filter='color like "red%" and likes > 50',
    output_fields=["color", "likes"],
    search_params={
        "hints": "iterative_filter"
    }
)

print("迭代过滤搜索结果:")
for hits in res:
    for hit in hits:
        print(f"ID: {hit['id']}, 距离: {hit['distance']:.4f}, 颜色: {hit['entity']['color']}, 点赞数: {hit['entity']['likes']}")

# 清理资源
client.drop_collection(collection_name)
