import gradio as gr
from src import cleaning



interface = gr.Interface(
    fn=cleaning.clean_text,
    inputs=gr.Textbox(label="Query Text"),
    outputs=gr.Textbox(label="Results"),
    title="Semantic Product Search",
    description="Enter query for semantic product search."
)
interface.launch()