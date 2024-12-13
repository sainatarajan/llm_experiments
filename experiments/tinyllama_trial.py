import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_weight_loading(model):
    """
    Test if the model's weights are loaded correctly.
    Prints the mean of each parameter to ensure they are not randomly initialized.
    """
    print("Testing weight loading...")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: mean = {param.data.mean().item():.6f}")


def run_llm_inference(prompt):
    # Check if CUDA (GPU) is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load a small, open-source LLM that can run on RTX 4060
    # We'll use TinyLlama, which is a compact model
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use half precision for better memory efficiency
        device_map="auto"  # Automatically distribute model across available GPU memory
    )
    print(model.__class__.__name__)
    # Test weight loading
    # test_weight_loading(model)

    exit()

    # Prepare the prompt
    # For chat models, we typically use a specific chat template
    chat_template = f"{prompt}"

    # Tokenize the input
    inputs = tokenizer(chat_template, return_tensors="pt").to(device)

    # Generate response
    outputs = model.generate(
        inputs.input_ids,
        max_length=500,  # Limit the response length
        num_return_sequences=1,
        do_sample=True,  # Use sampling for more creative outputs
        temperature=0.7,  # Control randomness
        top_p=0.9  # Nucleus sampling
    )

    # Decode the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


# Example usage
prompt = "Why can't black holes be created in a lab?"
result = run_llm_inference(prompt)
print("\nModel Response:")
print(result)