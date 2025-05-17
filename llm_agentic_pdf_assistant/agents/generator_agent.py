from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os


# Load the model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
local_path = "./local_model"

# Set device (use "mps" for Mac GPU, "cuda" for Nvidia, else "cpu")
device = "mps" if torch.backends.mps.is_available() else "cpu"


# Check if local model exists
if os.path.exists(local_path):
    print("✅ Loading model from local folder...")
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModelForCausalLM.from_pretrained(local_path)
else:
    print("⬇️  Downloading model from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Save locally for future use
    print("💾 Saving model locally...")
    tokenizer.save_pretrained(local_path)
    model.save_pretrained(local_path)



# Load the pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device= device
)

def generate_response(context, question):
    """
    Generate a response to a question based on the context using the TinyLlama model.
    
    Args:
        context (str): The context in which to generate the response.
        question (str): The question to answer.
    
    Returns:
        str: The generated response.
    """
    # Prepare the input for the model
    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
    
    # Generate the response
    response = generator(prompt, 
                         max_new_tokens=300, 
                         truncation = True, 
                         do_sample = True, 
                         temperature = 0.7, 
                         top_p = 0.95, 
                         num_return_sequences=1)
    
    # Extract and return the generated text
    return response[0]['generated_text'].split("A:")[-1].strip()

