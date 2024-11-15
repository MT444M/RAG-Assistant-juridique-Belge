import datetime
import pandas as pd

def clear_response():
    return ""

def copy_response(response):
    # Simulate copying to clipboard (can be integrated with clipboard library if needed)
    return response  # Placeholder action

def feedback_up(question, response, history_list):
    # Ajouter l'entrée question-réponse-date à l'historique
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    new_entry = [question, response, date]
    
    # Utilisation de pd.concat pour ajouter une nouvelle ligne au DataFrame
    history_list = pd.concat([history_list, pd.DataFrame([new_entry], columns=history_list.columns)], ignore_index=True)
    return history_list

def feedback_down():
    return """Nous nous excusons pour la réponse insatisfaisante.

Pouvez-vous clarifier davantage votre question pour que nous puissions mieux vous aider?

Merci de votre compréhension. Si nécessaire, essayez de modifier également les paramètres de recherches."""
