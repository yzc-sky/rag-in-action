from llama_index.readers.database import DatabaseReader

reader = DatabaseReader(
    scheme="mysql",
    host="localhost",
    port=3306,
    user="newuser",
    password="password",
    dbname="example_db"
)

query = "SELECT * FROM game_scenes" # 选择所有游戏场景 -> Text-to-SQL
documents = reader.load_data(query=query)

print(f"从数据库加载的文档数量: {len(documents)}")
print(documents)

# 参考文献
# https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/