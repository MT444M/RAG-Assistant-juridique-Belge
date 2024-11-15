from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
import torch
from config.config import DEFAULT_CONFIG

# Initialisation des modèles d'embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration du modèle BGEM3
if DEFAULT_CONFIG['MODEL_TYPE'] == 'bge_m3':
    bge_m3 = BGEM3FlagModel(DEFAULT_CONFIG['EMBEDDING_MODEL'], use_fp16=True, device=device)
    print("Initialisation du modèle BGEM3 :")
    print(f"  - Nom du modèle : {DEFAULT_CONFIG['EMBEDDING_MODEL']}")
    print(f"  - Périphérique utilisé : {device}")
    print(f"  - Précision mixte (half precision) activée : Oui\n")
    
# Configuration du modèle SentenceTransformer
else:
    model_bel = SentenceTransformer(DEFAULT_CONFIG['ST_MODEL_NAME'], trust_remote_code=True, device=device)
    print("Initialisation du modèle SentenceTransformer :")
    print(f"  - Nom du modèle : {DEFAULT_CONFIG['ST_MODEL_NAME']}")
    print(f"  - Périphérique utilisé : {device}\n")

def generate_embedding(query, model_type='bge_m3'):
    print(f"\nGénération de l'embedding pour la requête : {query}")
    print(f"  - Modèle sélectionné : {model_type}")
    
    if model_type == 'bge_m3':
        embedding = bge_m3.encode([query], batch_size=12, max_length=1024)["dense_vecs"]
        print(f"  - Embedding généré avec BGEM3")
    elif model_type == 'sentence_transformer':
        embedding = model_bel.encode([query])
        print(f"  - Embedding généré avec SentenceTransformer")
    else:
        raise ValueError("Model type non supporté")
    
    print(f"  - Taille de l'embedding : {len(embedding[0])}")
    return embedding[0]

# Ajoutez ce bloc pour obtenir des informations détaillées sur le GPU s'il est utilisé
if device == "cuda":
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"\nInformations sur le GPU :")
    print(f"  - Nom du GPU : {torch.cuda.get_device_name(0)}")
    print(f"  - Mémoire totale : {gpu_memory / (1024 ** 3):.2f} Go")
    print(f"  - Précision mixte activée : Oui\n")
else:
    print("\nLe modèle est exécuté sur le CPU.\n")

