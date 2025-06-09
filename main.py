from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
from huggingface_hub import login

app = Flask(__name__)

# Global objects to persist after startup
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Load Model Once on Startup
# --------------------------
def load_model_once():
    global model, tokenizer
    print("🔧 Loading model...")

    # Log in to Hugging Face Hub
    from dotenv import load_dotenv
    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))


# Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("NishKook/legal-qa-lora", token=True)

    # Load base Mistral model (no quantization)
    base_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype=torch.float32,     # Change to float32 for CPU compatibility
        device_map={"": device},       # Explicit device map
    )

    # Load LoRA adapter
    model_peft = PeftModel.from_pretrained(
        base_model,
        "NishKook/legal-qa-lora",
        torch_dtype=torch.float32,
        device_map={"": device},
        use_auth_token=True
    )

    model_peft.eval()
    model = model_peft
    print("✅ Model loaded.")

# --------------------------
# /warmup endpoint → 200 if model is loaded
# --------------------------
@app.route("/warmup", methods=["GET"])
def warmup():
    if model and tokenizer:
        return jsonify({"status": "READY"}), 200
    else:
        return jsonify({"status": "NOT READY"}), 503

# --------------------------
# Helper: Extract all text from PDF
# --------------------------
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

# --------------------------
# Helper: Build FAISS index on the fly from full PDF text
# --------------------------
def build_vector_index(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=384, chunk_overlap=32)
    chunks = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
    return FAISS.from_documents(chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

# --------------------------
# Helper: Return relevant context for a question
# --------------------------
def get_context(question, vectordb, k=7):
    results = vectordb.similarity_search(question, k=k)
    return "\n".join([doc.page_content for doc in results])

# --------------------------
# Helper: Generate answer using model + context
# --------------------------
def generate_answer(question, context):
    prompt = f"### Question:\n{question}\n\n### Context:\n{context}\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

# --------------------------
# /ask endpoint → Accept PDF path + question, return answer
# --------------------------
@app.route("/ask", methods=["POST"])
def ask():
    pdf_path = request.form.get("pdf_path")
    question = request.form.get("question")

    if not pdf_path or not os.path.exists(pdf_path):
        return jsonify({"error": "PDF not found"}), 404

    # Read and process the PDF
    full_text = extract_text_from_pdf(pdf_path)

    # Build vector index on-the-fly
    vectordb = build_vector_index(full_text)

    # Retrieve context using question
    context = get_context(question, vectordb)

    # Generate and return answer
    answer = generate_answer(question, context)
    return jsonify({"answer": answer})

# --------------------------
# Run the server
# --------------------------
if __name__ == "__main__":
    load_model_once()
    app.run(host="0.0.0.0", port=8000)
