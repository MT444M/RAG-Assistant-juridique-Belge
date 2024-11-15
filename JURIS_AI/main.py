import torch
torch.cuda.empty_cache()
torch.cuda.memory_allocated()



from interface.ui_components import create_legal_assistant_interface
from retrieval.search import search_documents
from retrieval.reranking import rerank_documents
from models.embeddings_processing import process_questions_and_get_embeddings
from generation.prompts import create_multi_query_prompt, create_hyde_prompt, create_generation_prompt
from generation.generator import generate_legal_response, generate_text
from config.config import DEFAULT_CONFIG
from models.llm import load_llm_model
from models.embeddings import bge_m3
from interface.button_actions import clear_response, copy_response, feedback_up, feedback_down

import gradio as gr

# Initialisation des composants de l'interface
demo, interface_components = create_legal_assistant_interface()

# Chargement du modèle LLM
llm_model, tokenizer = load_llm_model(
    model_id=DEFAULT_CONFIG['LLM_MODEL'], 
    quantized=DEFAULT_CONFIG['QUANTIZED']
)

def handle_submission(
    question=DEFAULT_CONFIG["DEFAULT_QUESTION"],
    multi_query=False,
    hyde=False,
    temperature=DEFAULT_CONFIG["TEMPERATURE"],
    top_p=DEFAULT_CONFIG["TOP_P"],
    max_tokens=DEFAULT_CONFIG["MAX_NEW_TOKENS"],
    num_docs=DEFAULT_CONFIG["RERANK_TOP_K"],
    show_refs=True,
    progress=gr.Progress(track_tqdm=True)
):
    progress(0, desc="Chargement des modèles (Encodeur & LLM)")

    # Déterminer la méthode d'analyse à utiliser
    method = 'multiqueries' if multi_query else 'hyde' if hyde else 'simple'

    # Générer un prompt basé sur les options multi_query ou HyDE
    if multi_query:
        prompt = create_multi_query_prompt(question)
        progress(0.20, desc="Génération des multi-queries")
        generated_text = generate_text(llm_model, tokenizer, prompt)
    elif hyde:
        prompt = create_hyde_prompt(question)
        progress(0.20, desc="Génération de la réponse HyDE")
        generated_text = generate_text(llm_model, tokenizer, prompt)
    else:
        prompt = question  # Utiliser directement la question si ni multi_query ni HyDE
        generated_text = None
        progress(0.20, desc="Utilisation de la question de base uniquement")

    # Processer les questions et obtenir les embeddings en fonction de la méthode choisie
    progress(0.40, desc="Obtention des embeddings")
    embeddings = process_questions_and_get_embeddings(
        base_question=question, generated_text=generated_text, method=method
    )

    # Effectuer la recherche initiale avec les embeddings générés
    progress(0.60, desc="Recherche des documents et Reranking")
    search_results = search_documents(
        embeddings
    )

    # Appliquer le reranking et obtenir les meilleurs articles
    top_articles = rerank_documents(
        question, search_results, bge_m3=bge_m3, num_docs=num_docs, max_length=1024,
        w_d=DEFAULT_CONFIG["WEIGHT_DENSE"], w_s=DEFAULT_CONFIG["WEIGHT_SPARSE"], w_c=DEFAULT_CONFIG["WEIGHT_COLBERT"]
    )

    # Génération de la réponse juridique
    progress(0.80, desc="Génération de la réponse finale")
    prompt_template = create_generation_prompt(documents=top_articles, question=question)
    response = generate_legal_response(
        llm_model=llm_model, tokenizer=tokenizer, prompt_template=prompt_template,
        documents=top_articles, question=question,
        temperature=temperature, top_p=top_p, max_new_tokens=max_tokens
    )

    # Préparer les articles et références pour l'affichage
    # Transformer les articles en liste de listes pour Gradio Dataframe
    formatted_sources = [[article, source] for article, source in top_articles] if show_refs else None

    return response, formatted_sources

def regenerate_response(question, multi_query, hyde, temperature, top_p, max_tokens, num_docs, show_refs, progress=gr.Progress(track_tqdm=True)):
    return handle_submission(question, multi_query, hyde, temperature, top_p, max_tokens, num_docs, show_refs, progress)


# Définition de l'interface utilisateur dans un bloc Gradio
with gr.Blocks() as demo:
    # Créez les composants d'interface
    _, interface_components = create_legal_assistant_interface()

    # Lier la fonction backend à l'action de soumission
    interface_components["submit_btn"].click(
        fn=handle_submission,
        inputs=[
            interface_components["question"], 
            interface_components["multiqueries"], 
            interface_components["hyde"], 
            interface_components["temperature"], 
            interface_components["top_p"], 
            interface_components["max_tokens"], 
            interface_components["num_docs"], 
            interface_components["show_refs"]
        ],
        outputs=[
            interface_components["response_box"], 
            interface_components["sources_box"]
        ]
    )

    # Boutons supplémentaires avec des vérifications pour `history_list`
    interface_components["clear_btn"].click(fn=clear_response, inputs=None, outputs=interface_components["response_box"]) 
    interface_components["copy_btn"].click(fn=copy_response, inputs=interface_components["response_box"], outputs=None)
    
    if "history_list" in interface_components:
        interface_components["feedback_up"].click(
            fn=feedback_up, 
            inputs=[interface_components["question"], interface_components["response_box"], interface_components["history_list"]],
            outputs=interface_components["history_list"]
        )
    
    interface_components["feedback_down"].click(
        fn=feedback_down, 
        inputs=None, 
        outputs=interface_components["response_box"]
    )

    # Ajouter le bouton "Régénérer"
    interface_components["regenerate_btn"].click(
        fn=regenerate_response,
        inputs=[
            interface_components["question"],
            interface_components["multiqueries"],
            interface_components["hyde"],
            interface_components["temperature"],
            interface_components["top_p"],
            interface_components["max_tokens"],
            interface_components["num_docs"],
            interface_components["show_refs"]
        ],
        outputs=[
            interface_components["response_box"],
            interface_components["sources_box"]
        ]
    )
    
# Lancez l'application Gradio
if __name__ == "__main__":
    demo.queue().launch()
