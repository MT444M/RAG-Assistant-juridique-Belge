import gradio as gr

# Demo de l'interface gradio sans fonction backend

# Exemple de fonction backend à adapter avec ta propre logique RAG
def generate_response(question, multi_query, hyde, temperature, top_p, max_tokens, num_docs, show_refs):
    response = f"Ceci est une réponse générée pour la question : {question}\n"
    response += "Options:\n"
    response += f" - Multi-queries: {multi_query}\n"
    response += f" - HyDE: {hyde}\n"
    response += f" - Temperature: {temperature}\n"
    response += f" - Top-p: {top_p}\n"
    response += f" - Max tokens: {max_tokens}\n"
    response += f" - Nombre de documents: {num_docs}\n"
    response += f" - Afficher les références: {show_refs}\n"
    # Ajouter la logique pour générer une réponse et les documents sources ici
    return response, [["Document1", "Référence1"], ["Document2", "Référence2"]] if show_refs else []

def create_legal_assistant_interface():
    interface_components = {}

    with gr.Blocks(theme="soft") as demo:
        gr.Markdown("# Assistant Juridique Belge")
        
        with gr.Row():
            # Panel Principal
            with gr.Column(scale=3):
                interface_components["question"] = gr.Textbox(
                    label="Votre question juridique",
                    placeholder="Posez votre question ici...",
                    lines=3
                )
                
                interface_components["submit_btn"] = gr.Button("Obtenir une réponse")
                
                with gr.Accordion("Documents sources", open=False):
                    interface_components["sources_box"] = gr.Dataframe(
                        headers=["Source", "Référence"],
                        label="Documents utilisés"
                    )
                
                interface_components["response_box"] = gr.Textbox(
                    label="Réponse",
                    lines=10
                )
                
                with gr.Row():
                    interface_components["copy_btn"] = gr.Button("📋 Copier")
                    interface_components["export_btn"] = gr.Button("📥 Exporter")
                    with gr.Row():
                        interface_components["feedback_up"] = gr.Button("👍")
                        interface_components["feedback_down"] = gr.Button("👎")
            
            # Panel Settings
            with gr.Column(scale=1):
                with gr.Accordion("Paramètres", open=True):
                    gr.Markdown("### Méthode de recherche")
                    interface_components["multi_query"] = gr.Checkbox(label="Multi-queries")
                    interface_components["hyde"] = gr.Checkbox(label="HyDE")
                    
                    gr.Markdown("### Paramètres du modèle")
                    interface_components["temperature"] = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        label="Temperature"
                    )
                    interface_components["top_p"] = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.9,
                        label="Top-p"
                    )
                    interface_components["max_tokens"] = gr.Slider(
                        minimum=100,
                        maximum=500,
                        value=250,
                        step=50,
                        label="Max tokens"
                    )
                    interface_components["num_docs"] = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Nombre de documents"
                    )
                    
                    interface_components["show_refs"] = gr.Checkbox(
                        label="Afficher les références",
                        value=True
                    )
        
        # Historique (Tab séparé)
        with gr.Tab("Historique"):
            interface_components["history_list"] = gr.Dataframe(
                headers=["Question", "Réponse", "Date"],
                label="Historique des questions"
            )

        # Lier les actions
        interface_components["submit_btn"].click(
            generate_response,
            inputs=[
                interface_components["question"], interface_components["multi_query"], interface_components["hyde"], 
                interface_components["temperature"], interface_components["top_p"], interface_components["max_tokens"], 
                interface_components["num_docs"], interface_components["show_refs"]
            ],
            outputs=[
                interface_components["response_box"], interface_components["sources_box"]
            ]
        )

    return demo, interface_components

demo, interface_components = create_legal_assistant_interface()
demo.launch()
