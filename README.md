# Fine-Tuning a Domain-Specific AI Model for Finance & Tech Industries

This project demonstrates how to fine-tune an open-source Large Language Model (Mistral/Llama) for industry-specific vocabulary and responses, focusing on Finance and Tech domains. It is a complete end-to-end full-stack AI project using PEFT/LoRA, BitsAndBytes 4-bit quantization, FastAPI, and React.

## Folder Structure

- `backend/`: Contains the datasets, the fine-tuning script (`train.py`), and the FastAPI server (`app.py`).
- `frontend/`: Contains the React + Vite application for the chatbot UI.

## 1. Backend Setup

### Prerequisites
Create a Python virtual environment and activate it:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Fine-Tuning the Model
The script `train.py` fine-tunes `mistralai/Mistral-7B-v0.1` on the `finance_dataset.json` and `tech_dataset.json`.
**Note:** This requires a GPU (recommended: NVIDIA RTX 3060+ with 16GB+ VRAM).

```bash
python train.py
```
This will save the model to `./fine_tuned_model`.

### Running the API Server
Start the FastAPI backend:
```bash
uvicorn app:app --reload
```
API URL: `http://127.0.0.1:8000`
Swagger Docs: `http://127.0.0.1:8000/docs`

---

## 2. Frontend Setup

### Install Dependencies
Navigate to the `frontend` directory and install the necessary packages.
```bash
cd frontend
npm install
npm install axios
```

### Running the Frontend
Start the Vite development server:
```bash
npm run dev
```

Navigate to `http://localhost:5173` (or the URL provided by Vite) in your browser.

## Sample Questions
### Finance
- Explain mutual funds
- What is EBITDA?
- Explain debt-to-equity ratio
- What is market capitalization?

### Tech
- Explain Kubernetes
- What is vector database?
- Difference between REST and GraphQL
- Explain microservices
