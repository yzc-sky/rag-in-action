# 安装依赖：pip install pymilvus
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType

# ——————————————
# 0. 连接 Milvus
# ——————————————
client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
print("✓ 已连接 Milvus 接口")

# ——————————————
# 1. 定义 Schema，显式添加 vector + color 字段
# ——————————————
fields = [
    FieldSchema(name="id",     dtype=DataType.INT64,        is_primary=True, auto_id=False),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=5),
    FieldSchema(name="color",  dtype=DataType.VARCHAR,      max_length=128),
]
schema = CollectionSchema(
    fields=fields,
    description="示例：向量字段 + color 倒排索引",
    enable_dynamic_field=False
)

collection_name = "color_demo"

# ——————————————
# 2. 创建集合
# ——————————————
client.create_collection(collection_name=collection_name, schema=schema)
print("✓ 已创建集合", collection_name)

# ——————————————
# 3. 插入数据
# ——————————————
data = [
    {"id": 0, "vector": [0.3, -0.6, 0.18, -0.26, 0.90], "color": "pink_8682"},
    {"id": 1, "vector": [0.19, 0.06, 0.70, 0.26, 0.84], "color": "red_7025"},
]
res = client.insert(collection_name=collection_name, data=data)
print("✓ 已插入数据：", res)

# ——————————————
# 4. 准备索引参数
# ——————————————
# 4.1 调用 prepare_index_params() 获取 IndexParams 对象 :contentReference[oaicite:0]{index=0}
index_params = client.prepare_index_params()

# 4.2 为向量字段添加 IVF_FLAT 索引 :contentReference[oaicite:1]{index=1}
index_params.add_index(
    field_name="vector",
    index_type="IVF_FLAT",
    metric_type="L2",
    params={"nlist": 128}
)

# 4.3 为 color 字段添加倒排索引
index_params.add_index(
    field_name="color",
    index_type="INVERTED",
    index_name="color_inverted_idx"
)

# ——————————————
# 5. 创建所有索引
# ——————————————
client.create_index(
    collection_name=collection_name,
    index_params=index_params
)
print("✓ 已为 vector & color 创建索引")

# ——————————————
# 6. 刷新并加载集合
# ——————————————
# client.flush([collection_name])
client.load_collection(collection_name=collection_name)
print("✓ 已 flush 且 load 集合")

# ——————————————
# 7. 演示查询和搜索
# ——————————————
# 7.1 普通按 color 过滤查询
qs = client.query(
    collection_name=collection_name,
    filter='color like "pink%"',
    output_fields=["id", "color"]
)
print("普通查询（pink*）：", qs)

# 7.2 带 color 过滤的向量搜索
query_vec = [0.3, -0.6, 0.18, -0.26, 0.90]
sr = client.search(
    collection_name=collection_name,
    data=[query_vec],
    anns_field="vector",
    param={"metric_type": "L2"},
    limit=3,
    filter='color like "red%"',
    output_fields=["color"]
)
print("向量搜索（red*）：", sr)

# ——————————————
# 8. 清理集合
# ——————————————
client.drop_collection(collection_name=collection_name)
print("✓ 已删除集合", collection_name)
