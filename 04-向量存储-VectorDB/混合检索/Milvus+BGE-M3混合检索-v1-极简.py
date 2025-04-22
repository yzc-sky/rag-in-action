import json
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker
)

# 1. 加载数据
with open("90-文档-Data/灭神纪/战斗场景.json", 'r', encoding='utf-8') as f:
    dataset = json.load(f)

docs = []
metadata = []
for item in dataset['data']:
    text_parts = [item['title'], item['description']]
    if 'combat_details' in item:
        text_parts.extend(item['combat_details'].get('combat_style', []))
        text_parts.extend(item['combat_details'].get('abilities_used', []))
    if 'scene_info' in item:
        text_parts.extend([
            item['scene_info'].get('location', ''),
            item['scene_info'].get('environment', ''),
            item['scene_info'].get('time_of_day', '')
        ])
    docs.append(' '.join(filter(None, text_parts)))
    metadata.append(item)

# 2. 生成向量
ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
docs_embeddings = ef(docs)

# 3. 连接Milvus
collection_name = "wukong_hybrid"
connections.connect(uri="./wukong.db")

# 4. 创建集合
fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="environment", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=ef.dim["dense"])
]

schema = CollectionSchema(fields)
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)

collection = Collection(name=collection_name, schema=schema, consistency_level="Strong")
collection.create_index("sparse_vector", {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"})
collection.create_index("dense_vector", {"index_type": "AUTOINDEX", "metric_type": "IP"})
collection.load()

# 5. 插入数据
batch_size = 50
for i in range(0, len(docs), batch_size):
    end_idx = min(i + batch_size, len(docs))
    batch_data = []
    for j in range(i, end_idx):
        item = metadata[j]
        batch_data.append({
            "text": docs[j],
            "id": item["id"],
            "title": item["title"],
            "category": item["category"],
            "location": item.get("scene_info", {}).get("location", ""),
            "environment": item.get("scene_info", {}).get("environment", ""),
            "sparse_vector": docs_embeddings["sparse"]._getrow(j),
            "dense_vector": docs_embeddings["dense"][j]
        })
    collection.insert(batch_data)

# 6. 混合搜索
def hybrid_search(query, category=None, environment=None, limit=5, weights={"sparse": 0.7, "dense": 1.0}):
    query_embeddings = ef([query])
    
    # 构建过滤表达式
    conditions = []
    if category:
        conditions.append(f'category == "{category}"')
    if environment:
        conditions.append(f'environment == "{environment}"')
    expr = " && ".join(conditions) if conditions else None
    
    search_params = {"metric_type": "IP", "params": {}}
    if expr:
        search_params["expr"] = expr
    
    dense_req = AnnSearchRequest(
        data=[query_embeddings["dense"][0]],
        anns_field="dense_vector",
        param=search_params,
        limit=limit
    )
    sparse_req = AnnSearchRequest(
        data=[query_embeddings["sparse"]._getrow(0)],
        anns_field="sparse_vector",
        param=search_params,
        limit=limit
    )
    rerank = WeightedRanker(weights["sparse"], weights["dense"])
    
    results = collection.hybrid_search(
        reqs=[dense_req, sparse_req],
        rerank=rerank,
        limit=limit,
        output_fields=["text", "id", "title", "category", "location", "environment"]
    )[0]
    
    return results