import torch

def rerank_documents(question, search_results, bge_m3, num_docs=3, max_length=1024, w_d=0.4, w_s=0.2, w_c=0.4, batch_size=1):
    # Extraire les textes des articles
    articles = [result["entity"]["chunk_article"] for result in search_results[0]]
    references = [result["entity"]["reference"] for result in search_results[0]]
    
    # Créer les paires question-article
    sentence_pairs = [[question, article] for article in articles]
    
    # Initialiser une liste pour stocker les scores
    all_scores = []
    
    # Calculer les scores par lots pour réduire la consommation de mémoire
    for i in range(0, len(sentence_pairs), batch_size):
        batch_pairs = sentence_pairs[i:i + batch_size]
        
        batch_scores = bge_m3.compute_score(
            batch_pairs,
            max_passage_length=max_length,
            weights_for_different_modes=[w_d, w_s, w_c]  # [dense, sparse, colbert]
        )
        
        # Ajouter les scores du batch courant aux scores totaux
        all_scores.extend(batch_scores['colbert+sparse+dense'])
        
        # Libérer la mémoire GPU pour le batch traité
        torch.cuda.empty_cache()
    
    # Combiner les scores avec les articles et les références
    ranked_results = [
        (article, reference, score)
        for article, reference, score in zip(articles, references, all_scores)
    ]
    
    # Trier les résultats par score
    ranked_results.sort(key=lambda x: x[-1], reverse=True)
    
    # Retourner les `num_docs` meilleurs articles
    top_articles = [(article, reference) for article, reference, _ in ranked_results[:num_docs]]
    
    return top_articles
