from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv


load_dotenv()

# Create models directory if it doesn't exist
if not os.path.exists("models"):
    os.makedirs("models")

token = os.getenv("HUGGINGFACE_TOKEN")
model_id = "google/gemma-3-1b-it"
cache_dir = "./models"

print(f"Downloading/Loading model '{model_id}' into '{cache_dir}'...")

# Load tokenizer and model, specifying the cache directory
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_id, token=token, cache_dir=cache_dir)

print("Model loaded successfully.")