import gradio as gr
from config.config import DEFAULT_CONFIG

def create_legal_assistant_interface():
    interface_components = {}

    with gr.Blocks(theme="soft") as demo:
        gr.Markdown("# ⚖️ <span style='font-weight: bold; color: #ff7b00;'>JURIS_AI</span>: <span style='font-weight: normal;'>Assistant Juridique Belge</span>")
                   
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

                # interface_components["progress_bar"] = gr.Progress()

                with gr.Row(): 
                    interface_components["clear_btn"] = gr.Button("🗑️ Clear")
                    interface_components["regenerate_btn"] = gr.Button("🔄 Regenerate")

                
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
                    interface_components["multiqueries"] = gr.Checkbox(label="Multi-queries")
                    interface_components["hyde"] = gr.Checkbox(label="HyDE")
                    
                    gr.Markdown("### Paramètres du modèle")
                    interface_components["temperature"] = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value= DEFAULT_CONFIG["TEMPERATURE"],
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
                        maximum=1000,
                        value= DEFAULT_CONFIG["MAX_NEW_TOKENS"],
                        step=50,
                        label="Max tokens"
                    )
                    interface_components["num_docs"] = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value= DEFAULT_CONFIG["RERANK_TOP_K"],
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
        
        gr.Image("interface/logo.png", elem_id="logo", width=150)

    return demo, interface_components
