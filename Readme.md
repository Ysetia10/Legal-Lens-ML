# LegalLens ML API

LegalLens is an intelligent question-answering system designed to help users understand legal documents. Users can upload a legal PDF and ask questions related to its content. The system uses a vector-based search for context retrieval and a fine-tuned Mistral-7B model with LoRA for answer generation.

---

# Features

- 🔍 Extracts and indexes text from legal PDFs.
- 🧠 Retrieves relevant context using FAISS + Hugging Face embeddings.
- 🤖 Generates precise legal answers using a fine-tuned Mistral-7B model with LoRA.
- 🌐 Exposes REST API endpoints via Flask.