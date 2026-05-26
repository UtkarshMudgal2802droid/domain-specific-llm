import os
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FINE_TUNED_MODEL = "./fine_tuned_model"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

model_name = FINE_TUNED_MODEL if os.path.exists(FINE_TUNED_MODEL) else BASE_MODEL

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
)
model.eval()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_text(data: PromptRequest):
    messages = [
        {"role": "system", "content": "You are an expert Finance and Tech AI assistant."},
        {"role": "user", "content": data.prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "assistant" in text.lower():
        return {"response": text}
    return {"response": text.strip()}
