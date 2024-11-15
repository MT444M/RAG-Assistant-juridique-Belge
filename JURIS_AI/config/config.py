# Configuration générale
DEFAULT_CONFIG = {
    # Modèles
    "EMBEDDING_MODEL": "BAAI/bge-m3",  # Modèle d'embeddings default
    # Paramètres model embeddings sentence transformer
    "ST_MODEL_NAME": "all-MiniLM-L6-v2",
    "MODEL_TYPE": "bge_m3",

    "LLM_MODEL": "meta-llama/Llama-3.2-3B-Instruct",  # Modèle de LLM default
    "QUANTIZED": True,
    
    # Milvus Configuration
    "MILVUS_HOST": "localhost",  # Hôte Milvus
    "MILVUS_PORT": 19530,  # Port Milvus
    "COLLECTION_NAME": "RSchunk_articles_collection",  # Nom de la collection Milvus
    
    # Paramètres de recherche
    "DEFAULT_TOP_K": 20,  # Nombre de résultats de recherche par défaut
    "RERANK_TOP_K": 3,  # Nombre de résultats après reranking
    
    # Paramètres de génération
    "MAX_NEW_TOKENS": 400,  # Nombre maximal de nouveaux tokens générés
    "TEMPERATURE": 0.5,  # Température pour la génération de texte
    "TOP_P": 0.3,  # Top-p pour la génération de texte

    # Paramètres RERANKING
    "WEIGHT_DENSE": 0.4,
    "WEIGHT_SPARSE": 0.2,
    "WEIGHT_COLBERT": 0.4,
    
    # Paramètres des méthodes avancées
    "MULTI_QUERIES_ENABLED": False,  # Activation de multi-queries
    "HYDE_ENABLED": False,  # Activation de HyDE
    "ADVANCED_METHOD":"simple",

    # Valeur par défaut pour la question utilisateur
    "DEFAULT_QUESTION": ""  # Question par défaut pour l'initialisation

    
}

# Vérification rapide des paramètres
if __name__ == "__main__":
    for key, value in DEFAULT_CONFIG.items():
        print(f"{key}: {value}")
