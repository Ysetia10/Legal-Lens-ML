from fastapi import FastAPI, UploadFile, File
# from app.model import load_model, predict

app = FastAPI()
# model = load_model("models/your_model.pkl")

@app.post("/api/analyse")
async def analyse(file: UploadFile = File(...)):
    content = await file.read()
    # result = predict(model, content)
    # return {"result": result}
