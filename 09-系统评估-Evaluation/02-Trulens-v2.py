import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI as OpenAIClient  # 避免冲突
from trulens.core import TruSession, Feedback, Select
from trulens.apps.app import TruApp, instrument
from trulens.providers.openai import OpenAI as TruLensOpenAI
import numpy as np

# Set API Key
# os.environ["OPENAI_API_KEY"] = "your_key_here"

# Embedding initialization
embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get("OPENAI_API_KEY"),
                                             model_name="text-embedding-ada-002")
chroma_client = chromadb.Client()
vector_store = chroma_client.get_or_create_collection("Info", embedding_function=embedding_function)

# add data
vector_store.add("starbucks_info", documents=[
    """
    Starbucks Corporation is an American multinational chain of coffeehouses headquartered in Seattle, Washington.
    As the world's largest coffeehouse chain, Starbucks is seen to be the main representation of the United States' second wave of coffee culture.
    """
])

class RAG:
    @instrument
    def retrieve(self, query: str):
        results = vector_store.query(query_texts=[query], n_results=2)
        return results["documents"][0] if results["documents"] else []

    @instrument
    def generate_completion(self, query: str, context: list):
        oai_client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY"))
        context_str = "\n".join(context) if context else "No context available."
        completion = oai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Context: {context_str}\nQuestion: {query}"}]
        ).choices[0].message.content
        return completion

    @instrument
    def query(self, query: str):
        context = self.retrieve(query)
        return self.generate_completion(query, context)

# TruLens initialization
session = TruSession(database_redact_keys=True)
session.reset_database()

# TruLens OpenAI provider
provider = TruLensOpenAI(model_engine="gpt-4")

# Feedback functions
f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness") \
    .on(Select.RecordCalls.retrieve.rets).on_output()
f_answer_relevance = Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance") \
    .on_input().on_output()
f_context_relevance = Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance") \
    .on_input().on(Select.RecordCalls.retrieve.rets[:]).aggregate(np.mean)

# TruApp setup
rag = RAG()
tru_rag = TruApp(
    rag,
    app_name="RAG",
    app_version="base",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance]
)

# Run query
with tru_rag as recording:
    response = rag.query("What wave of coffee culture is Starbucks seen to represent in the United States?")
    print(f"Response: {response}")

# View results
print(session.get_leaderboard())
