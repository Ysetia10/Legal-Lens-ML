import pickle
import io

def load_model(path: str):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def predict(model, file_bytes: bytes):
    # You can adjust this part based on your actual preprocessing
    # For now, we assume it's a text-based document like .txt or .docx
    try:
        text = file_bytes.decode("utf-8", errors="ignore")
        # Convert to model-compatible input if needed
        # Example: result = model.predict([text])
        result = model.predict([text])[0]
        return result
    except Exception as e:
        return f"Prediction failed: {str(e)}"
