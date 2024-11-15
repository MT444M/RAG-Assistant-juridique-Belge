from retrieval.vector_store import connect_to_milvus, load_collection, list_partitions, load_partitions
from models.embeddings_processing import process_questions_and_get_embeddings
from pymilvus import MilvusClient
from config.config import DEFAULT_CONFIG

def search_documents(embeddings_query):
    # Connect to Milvus and load the collection
    connect_to_milvus()
    collection_name = DEFAULT_CONFIG["COLLECTION_NAME"]
    collection = load_collection(collection_name)
    
    client = MilvusClient(uri=f"http://{DEFAULT_CONFIG['MILVUS_HOST']}:{DEFAULT_CONFIG['MILVUS_PORT']}")
    
    res = list_partitions(client, collection_name)
    selected_partitions = res[1:]  # Excluding the default partition
    
    load_partitions(client, collection_name, selected_partitions)
        
    search_results = client.search(
        collection_name=collection_name,
        data=embeddings_query,
        partition_names=selected_partitions,
        limit=DEFAULT_CONFIG["DEFAULT_TOP_K"],
        search_params={"metric_type": "COSINE", "params": {}},
        output_fields=['reference', 'article_id', 'chunk_article']
    )
    
    return search_results
