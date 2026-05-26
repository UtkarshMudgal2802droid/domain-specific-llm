from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
import torch

# -----------------------------------
# LOAD MODEL
# -----------------------------------
model_path = "./fine_tuned_model"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
except Exception as e:
    print(f"Warning: Model not found at {model_path}. Please train the model first.")
    tokenizer = None
    model = None

# -----------------------------------
# FASTAPI APP
# -----------------------------------
app = FastAPI()

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# REQUEST MODEL
# -----------------------------------
class PromptRequest(BaseModel):
    prompt: str

# -----------------------------------
# GENERATE ENDPOINT
# -----------------------------------
@app.post("/generate")
async def generate_text(data: PromptRequest):
    if not model or not tokenizer:
        return {"response": "Model not loaded. Please ensure the model is trained and exists in './fine_tuned_model'."}

    inputs = tokenizer(
        data.prompt,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )
    
    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
    
    return {
        "response": response
    }
