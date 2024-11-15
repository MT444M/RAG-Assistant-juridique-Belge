def create_multi_query_prompt(question):
    return f"""À partir de la question de base '{question}', 
génère 3 nouvelles questions connexes qui explorent différents aspects ou angles du sujet.
Assure-toi que chaque nouvelle question apporte une perspective unique ou approfondit un élément spécifique de la question initiale.
Donne uniquement les questions, sans explications ni commentaires supplémentaires.
Format de réponse attendu :
1. [Question 1]
2. [Question 2]
3. [Question 3]
"""

def create_hyde_prompt(question):
    return f"""Génère une réponse hypothétique concise mais informative à la question suivante :

Question : {question}

Fournissez seulement une réponse qui pourrait plausiblement apparaître dans un document pertinent. La réponse doit être factuelle, directe et informative.

Réponse hypothétique :
"""

def create_generation_prompt(documents, question):
    return f"""[CONTEXTE]
Vous êtes un assistant juridique spécialisé dans le droit belge. Votre rôle est d'aider à comprendre et interpréter la législation belge en vous basant uniquement sur les documents fournis.

[INSTRUCTIONS]
1. Analysez attentivement les extraits de documents juridiques ci-dessous
2. Répondez à la question en utilisant UNIQUEMENT les informations présentes dans ces documents
3. Si certains aspects de la question ne peuvent pas être traités avec les documents fournis, indiquez-le clairement
4. Privilégiez une réponse concise et directe

[DOCUMENTS]
{documents}

[QUESTION]
{question}

[FORMAT DE RÉPONSE]
- Réponse directe à la question
- Citation systématique des références juridiques utilisées
- Si des informations manquent, le mentionner brièvement

[TONALITÉ]
- Professionnel mais accessible
- Factuel
- Précis

[RÉPONSE]
"""
