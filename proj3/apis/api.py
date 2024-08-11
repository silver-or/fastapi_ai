# STEP 1
from transformers import pipeline

from fastapi import FastAPI, Form

# STEP 2
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")
summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
question_answerer = pipeline("question-answering", model="stevhliu/my_awesome_qa_model")

app = FastAPI()

@app.post("/classification/")
async def cls(text: str = Form()):
    result = classifier(text)
    return {"result": result}

@app.post("/summarization/")
async def summarization(text: str = Form()):
    result = summarizer(text)
    return {"result": result}

@app.post("/qna/")
async def qna(question: str = Form(), context: str = Form()):
    result = question_answerer(question=question, context=context)
    return {"result": result}