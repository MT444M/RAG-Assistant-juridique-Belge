import torch
from config.config import DEFAULT_CONFIG

def generate_text(llm_model, tokenizer, prompt, max_new_tokens= DEFAULT_CONFIG['MAX_NEW_TOKENS']):
    """
    Génère du texte à partir d'un prompt donné en utilisant le modèle et le tokenizer spécifiés.
    
    Args:
        llm_llm_model: Le modèle LLM chargé
        tokenizer: Le tokenizer associé
        prompt: Le prompt à utiliser pour la génération
        max_new_tokens: Nombre maximum de tokens à générer
    
    Returns:
        str: Le texte généré
    """
    # Tokenisation du prompt, envoi sur GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Génération de nouvelles questions ou réponse HyDE
    out = llm_model.generate(
        **input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True  # Active le sampling pour plus de diversité
    )
    
    # Décodage uniquement de la partie générée (ignore le prompt d'entrée)
    generated_tokens = out[0][input_ids['input_ids'].shape[-1]:]  # On découpe pour ignorer le prompt
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text

def generate_legal_response(llm_model, tokenizer, prompt_template, documents, question, 
                            max_new_tokens=DEFAULT_CONFIG['MAX_NEW_TOKENS'], 
                            temperature=DEFAULT_CONFIG['TEMPERATURE'],
                            top_p=DEFAULT_CONFIG['TOP_P']):
    """
    Génère une réponse juridique basée sur les documents fournis.
    
    Args:
        llm_model: Le modèle LLM chargé
        tokenizer: Le tokenizer associé
        prompt_template: Le template du prompt avec {documents} et {question} comme placeholders
        documents: Liste de tuples (contenu, référence)
        question: Question de l'utilisateur
        max_new_tokens: Nombre maximum de tokens à générer
        temperature: Température pour le sampling
        top_p: Valeur pour le nucleus sampling
    
    Returns:
        str: Réponse générée
    """
    # Formatage des documents avec leurs références
    formatted_docs = []
    for i, (content, reference) in enumerate(documents, 1):
        formatted_doc = f"Document {i}:\nRéférence: {reference}\nContenu: {content.strip()}"
        formatted_docs.append(formatted_doc)
    
    formatted_docs = "\n\n".join(formatted_docs)
    
    # Construction du prompt final
    prompt = prompt_template.format(
        documents=formatted_docs,
        question=question
    )
    
    # Tokenisation et envoi sur GPU
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to("cuda")
    
    # Génération
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Décodage uniquement de la partie générée
    generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response.strip()
