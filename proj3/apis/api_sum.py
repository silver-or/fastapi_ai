# STEP 1
from transformers import pipeline

from fastapi import FastAPI, Form

# STEP 2
summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")

app = FastAPI()


@app.post("/summarization/")
async def summarization(text: str = Form()):
    # STEP 3: text
    # STEP 4
    result = summarizer(text)
    return {"result": result}