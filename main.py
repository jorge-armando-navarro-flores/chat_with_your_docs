
import gradio as gr

# local imports
from models.llms import OllamaModel
from controllers.chatbot_controller import ChatBotController

undo_button = gr.Button("↩️ Undo")
clear_button = gr.Button("🗑️  Clear")

chatbot = ChatBotController(OllamaModel("orca-mini"))


def filter_model_types(model_type):
    api_key_field = model_type != "Ollama"
    return gr.Text(placeholder="Input your API key", visible=api_key_field), gr.Dropdown(value=chatbot.model_types[model_type][0],
                                                                               choices=chatbot.model_types[model_type])

def filter_doc_types(doc_type):
    url_field = False
    file_field = False

    if doc_type == "PDF":
        file_field = True
    else:
        url_field = True
    return gr.Textbox("URL", visible=url_field), gr.File(visible=file_field)


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            model_label = gr.Label("Set your Model")
            model_type = gr.Radio(label="Model Source", value="Ollama", choices=["Ollama", "OpenAI", "HuggingFace"])
            selected_model = gr.Dropdown(label="Model Selection", value="orca-mini", choices=["orca-mini", "llama2:latest"])
            api_key = gr.Textbox(label="API key", type="password", visible=False)
            model_type.input(chatbot.set_model, inputs=[model_type, selected_model, api_key], outputs=[model_label])
            selected_model.input(chatbot.set_model, inputs=[model_type, selected_model, api_key], outputs=[model_label])
            api_key.input(chatbot.set_model, inputs=[model_type, selected_model, api_key], outputs=[model_label])
            model_type.change(filter_model_types, [model_type], outputs=[api_key, selected_model])
            doc_label = gr.Label("Set your Docs")

            doc_type = gr.Radio(label="Model Type", value="PDF", choices=["PDF", "WEB", "YouTube"])
            url = gr.Textbox(label="Document Source", placeholder="URL", visible=False)
            file = gr.File()
            doc_type.change(filter_doc_types, inputs=[doc_type], outputs=[url, file])

            process_button = gr.Button("Process")
            process_button.click(chatbot.set_retrieval, inputs=[doc_type, url, file], outputs=[doc_label])

        with gr.Column(scale=3):
            gr.ChatInterface(chatbot.predict, retry_btn="🔄  Retry", undo_btn=undo_button, clear_btn=clear_button)

if __name__ == "__main__":
    demo.launch()