from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
 
MODEL_ID = "mistralai/Mistral-7B-v0.1"
 
def load_model():
    print(f"Loading model: {MODEL_ID}")
 
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
 
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
 
    print("Model loaded successfully!")
    return model, tokenizer