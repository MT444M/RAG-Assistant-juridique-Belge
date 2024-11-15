import re
from models.embeddings import generate_embedding
from config.config import DEFAULT_CONFIG

def process_questions_and_get_embeddings(base_question, generated_text=None, method=DEFAULT_CONFIG["ADVANCED_METHOD"]):
    if method == 'hyde':
        # Extraire la réponse hypothétique
        hyde_response = generated_text.split("Réponse hypothétique :")[-1].strip()
        hyde_response = hyde_response.split("\n\n")[0].strip()  # Stopper après le premier double saut de ligne

        # Combiner la question de base avec la réponse hypothétique
        combined_text = f"Question : {base_question}\nRéponse : {hyde_response}"

        # Utiliser generate_embedding pour créer l'embedding
        embedding = generate_embedding(combined_text, model_type=DEFAULT_CONFIG["MODEL_TYPE"])
        
        return [embedding], combined_text  # Retourner une liste avec un seul embedding
    
    elif method == 'multiqueries':
        # Extraire les nouvelles questions
        new_questions = re.findall(r'\d+\.\s*(.*)', generated_text)

        # Combiner la question de base avec les nouvelles questions
        all_questions = [base_question] + new_questions

        # Créer les embeddings pour chaque question en utilisant generate_embedding
        embeddings = [generate_embedding(question, model_type=DEFAULT_CONFIG["MODEL_TYPE"]) for question in all_questions]
        
        return embeddings
    
    elif method == 'simple':
        # Utiliser generate_embedding pour créer l'embedding de la question de base
        embedding = generate_embedding(base_question, model_type=DEFAULT_CONFIG["MODEL_TYPE"])
        
        return [embedding]  # Retourner une liste avec un seul embedding

    else:
        raise ValueError("Méthode non supportée : doit être 'hyde', 'multiqueries', ou 'simple'")
