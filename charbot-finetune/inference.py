import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_DIR = "./mistral-persona-lora"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Set up text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

def chat(persona, message, max_tokens=200):
    prompt = f"instruction: {message}\ninput: persona: {persona}\noutput:"
    responses = generator(prompt, max_new_tokens=max_tokens, do_sample=True, top_p=0.95, temperature=0.8)
    return responses[0]['generated_text'].split("output:")[-1].strip()

# Example usage
if __name__ == "__main__":
    while True:
        persona = input("Choose persona (natural, friendly, sarcastic, professional, flirty): ").strip().lower()
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = chat(persona, user_input)
        print(f"{persona.title()} Bot:", response)
