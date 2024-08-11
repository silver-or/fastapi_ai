# STEP 1
from transformers import pipeline

from fastapi import FastAPI, Form

# STEP 2
question_answerer = pipeline("question-answering", model="stevhliu/my_awesome_qa_model")

app = FastAPI()


@app.post("/qna/")
async def qna(question: str = Form(), context: str = Form()):
    # STEP 3: text
    # STEP 4
    result = question_answerer(question=question, context=context)
    return {"result": result}