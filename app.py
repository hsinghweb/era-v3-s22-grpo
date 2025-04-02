import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from peft import PeftModel

# Load base model
BASE_MODEL = "microsoft/phi-2"
FINE_TUNED_MODEL = "phi2-qlora-grpo"

# Initialize tokenizer and models
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Load fine-tuned model
fine_tuned_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
fine_tuned_model = PeftModel.from_pretrained(fine_tuned_model, FINE_TUNED_MODEL)

def generate_response(prompt, model_choice, max_length=512, temperature=0.7):
    selected_model = base_model if model_choice == "Base Model" else fine_tuned_model
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(selected_model.device) for k, v in inputs.items()}
    
    outputs = selected_model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Create Gradio interface
with gr.Blocks(title="Phi-2 Model Comparison") as demo:
    gr.Markdown("# Phi-2 Base vs Fine-tuned Model Comparison")
    gr.Markdown("Compare responses between the base Phi-2 model and our GRPO fine-tuned version")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Enter your prompt", lines=3)
            model_choice = gr.Radio(
                choices=["Base Model", "Fine-tuned Model"],
                label="Select Model",
                value="Base Model"
            )
            max_length = gr.Slider(
                minimum=64,
                maximum=1024,
                value=512,
                step=64,
                label="Maximum Length"
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.7,
                step=0.1,
                label="Temperature"
            )
            submit_btn = gr.Button("Generate Response")
        
        with gr.Column():
            output = gr.Textbox(label="Generated Response", lines=10)
    
    submit_btn.click(
        generate_response,
        inputs=[prompt, model_choice, max_length, temperature],
        outputs=[output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()