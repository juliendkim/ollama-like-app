from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import sys
import traceback

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
# Configure logging to record server status and aid in debugging.
# All logs at DEBUG level and above are recorded in the 'server.log' file and the console.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("server.log"),  # Save logs to file
        logging.StreamHandler(sys.stdout)   # Output logs to terminal
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Model Path: Specifies the local directory where the Gemma 3 1B model is stored.
MODEL_PATH = "models/gemma3-1b-it"

def get_device():
    """
    Automatically selects the best available hardware accelerator (device).
    Priority: NVIDIA GPU (cuda) > Apple Silicon GPU (mps) > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"  # macOS Metal Performance Shaders (Apple Silicon)
    return "cpu"

DEVICE = get_device()

# Data Type (Dtype) Selection:
# Use float16 on GPU (cuda, mps) for memory efficiency and speed.
# Use float32 on CPU for compatibility.
DTYPE = torch.float16 if DEVICE in ["cuda", "mps"] else torch.float32

logger.info(f"Loading model from {MODEL_PATH} on {DEVICE} (DTYPE: {DTYPE})...")

# -----------------------------------------------------------------------------
# Model & Tokenizer Loading
# -----------------------------------------------------------------------------
try:
    # Load Tokenizer: A tool that converts text into numbers (tokens) the model can understand.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Load Model: Loads the actual language model into memory.
    # For Apple Silicon (mps), device_map="auto" might not always work reliably, so we assign it manually.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        device_map=DEVICE if DEVICE != "mps" else None, 
    )
    
    # If using MPS, explicitly move the model to the device.
    if DEVICE == "mps":
        model = model.to(DEVICE)
        
    logger.info(f"Model loaded successfully on {DEVICE}.")
except Exception as e:
    # Record a critical error and exit if model loading fails.
    logger.critical(f"Error loading model: {e}")
    logger.debug(traceback.format_exc())
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}")

# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------
# Define the data structure for client requests using Pydantic.

class Message(BaseModel):
    """Individual message structure for chat history"""
    role: str       # Message author (e.g., 'user', 'assistant')
    content: str    # Message content

class ChatRequest(BaseModel):
    """Structure for the full chat request"""
    message: str                # Current user message
    history: list[Message] = [] # Previous conversation history (default is an empty list)
    max_length: int = 200       # Maximum number of tokens to generate
    temperature: float = 0.7    # Control diversity (higher is more creative, lower is more consistent)

# -----------------------------------------------------------------------------
# Middleware
# -----------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all incoming HTTP requests and outgoing responses.
    Logs method, URL, and response status code.
    """
    logger.info(f"Incoming request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Request completed with status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {e}")
        # Do not swallow the exception; pass it to the next handler.
        return await call_next(request) 

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/")
async def root():
    """Default endpoint for Health Check"""
    logger.debug("Root endpoint called")
    return {"status": "ok", "model": "gemma3-1b-it"}

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat Endpoint:
    1. Receives user input and conversation history.
    2. Converts it into a prompt format understood by the model.
    3. Generates a response using the model.
    4. Decodes the generated tokens into text and returns it.
    """
    logger.info(f"Chat request received. Length: {len(request.message)}, History: {len(request.history)}, Max tokens: {request.max_length}")
    try:
        # 1. Construct messages: Previous history + current user message
        messages = []
        for msg in request.history:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add latest user message
        messages.append({"role": "user", "content": request.message})

        # 2. Apply chat template and tokenize
        if hasattr(tokenizer, "apply_chat_template"):
            logger.debug(f"Applying chat template with {len(messages)} messages...")
            # apply_chat_template adds model-specific special tokens (e.g., <start_of_turn>).
            input_data = tokenizer.apply_chat_template(
                messages, 
                return_tensors="pt", # Return as PyTorch tensors
                add_generation_prompt=True # Add prompt to induce assistant response
            )
            
            # Handle return type (dictionary or tensor)
            if isinstance(input_data, dict) or hasattr(input_data, "input_ids"):
                 input_ids = input_data["input_ids"]
            else:
                 input_ids = input_data

            # Move data to computation device (GPU/CPU)
            input_ids = input_ids.to(DEVICE)
        else:
            # Fallback for models without a chat template
            logger.debug("Encoding raw text (Template not found)...")
            context = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            input_ids = tokenizer.encode(context, return_tensors="pt").to(DEVICE)

        logger.debug(f"Input tensor shape: {input_ids.shape}")

        # 3. Generate text using the model (Inference)
        logger.debug("Generating response...")
        
        # Create attention mask (prevents warnings and improves accuracy when pad_token == eos_token)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,     # Explicitly pass attention mask
            max_new_tokens=request.max_length, # Maximum number of new tokens to generate
            temperature=request.temperature,   # Control probability distribution
            do_sample=True,                    # Use sampling (enable diversity)
            pad_token_id=tokenizer.eos_token_id # Set padding token to prevent errors
        )
        
        # 4. Decode result
        logger.debug("Decoding response...")
        # Extract only the newly generated part (after input_ids) and decode to text.
        response_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        
        logger.info("Response generated successfully.")
        return {"response": response_text}
    except Exception as e:
        # Record detailed logs and return 500 status code in case of error.
        logger.error(f"Error during chat generation: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Start server: 0.0.0.0 allows access from all network interfaces.
    uvicorn.run(app, host="0.0.0.0", port=8000)
