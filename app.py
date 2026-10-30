import gradio as gr

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 2. Lowercase
    text = text.lower()
    
    # 3. Normalize dimensions: '8 x 40' -> '8x40'
    text = re.sub(r'(\d)\s*x\s*(\d)', r'\1x\2', text)
    
    # 4. Strip 'mm' directly following a dimension: 'm8x40 mm' -> 'm8x40'
    text = re.sub(r'(x\d+(?:[.,]\d+)?)\s*mm\b', r'\1', text)
    
    # 5. Normalize multiple spaces to a single space
    return ' '.join(text.split())

interface = gr.Interface(
    fn=clean_text,
    inputs=gr.Textbox(label="Text to Clean"),
    outputs=gr.Textbox(label="Cleaned Text"),
    title="Text Cleaner",
    description="Enter some text and get it cleaned."
)
interface.launch()