from pymilvus import MilvusClient, Collection, connections
from config.config import DEFAULT_CONFIG

def connect_to_milvus():
    # Connexion à Milvus
    connections.connect("default", host=DEFAULT_CONFIG["MILVUS_HOST"], port=DEFAULT_CONFIG["MILVUS_PORT"])
    print(f"\nConnexion à Milvus réussie :")

def load_collection(collection_name):
    # Chargement de la collection
    collection = Collection(collection_name)
    collection.load()
    print(f"\nChargement de la collection Milvus : '{collection_name}'")
    print(f"  - Nombre de documents : {collection.num_entities}")
    return collection

def list_partitions(client, collection_name):
    # Liste des partitions de la collection
    partitions = client.list_partitions(collection_name=collection_name)
    return partitions

def load_partitions(client, collection_name, partition_names):
    # Chargement des partitions spécifiques
    client.load_partitions(collection_name=collection_name, partition_names=partition_names)
    print(f"\nChargement des partitions dans la collection '{collection_name}' :")
    print(f"\nPartitions dans la collection '{collection_name[:3]}' . . .:")

