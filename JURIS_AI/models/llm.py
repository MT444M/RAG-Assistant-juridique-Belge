import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from config.config import DEFAULT_CONFIG

def load_llm_model(model_id=DEFAULT_CONFIG['LLM_MODEL'], quantized=DEFAULT_CONFIG['QUANTIZED']):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nChargement du modèle de langage : {model_id}")
    print(f"  - Périphérique d'exécution : {device}")
    print(f"  - Quantisation activée : {'Oui' if quantized else 'Non'}")

    if quantized:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if device == "cuda" else torch.float16
        )
        llm_model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=quantization_config,
            device_map="cuda" if device == "cuda" else "cpu"
        )
        
        # Affichage des détails de quantification
        print("  - Configuration de quantification en 4 bits activée")
        print("  - Type de quantification : nf4")
        print(f"  - Type de calcul : {'bfloat16' if device == 'cuda' else 'float16'}")
    else:
        llm_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda" if device == "cuda" else "cpu")
        print("  - Modèle chargé sans quantification")

    # Chargement du tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("  - Tokenizer chargé")
    print(f"  - Vocabulaire du tokenizer : {len(tokenizer)} tokens\n")

    # Information sur la mémoire GPU si CUDA est utilisé
    if device == "cuda":
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"\nInformations sur le GPU :")
        print(f"  - Nom du GPU : {torch.cuda.get_device_name(0)}")
        print(f"  - Mémoire totale : {gpu_memory / (1024 ** 3):.2f} Go\n")
    else:
        print("\nLe modèle est exécuté sur le CPU.\n")

    return llm_model, tokenizer
