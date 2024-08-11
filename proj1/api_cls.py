# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

from PIL import Image
import io
import numpy as np

from fastapi import FastAPI, File, UploadFile

# STEP 2: Create an ImageClassifier object.
base_options = python.BaseOptions(model_asset_path='models\efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=4)
classifier = vision.ImageClassifier.create_from_options(options)

app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile): # meta data
    contents = await file.read()

    # STEP 3: Load the input image. (!!중요!!)
    # 1. create pil image from http file 
    # 1-1. convert http file to file (텍스트 파일을 기계어로 변환)
    pil_img = Image.open(io.BytesIO(contents))
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))
    # image = mp.Image.create_from_file("cat.jpg") # 로컬 파일

    # STEP 4: Classify the input image.
    classification_result = classifier.classify(image)

    # STEP 5: Process the classification result. In this case, visualize it.
    top_category = classification_result.classifications[0].categories[0]
    result = f"{top_category.category_name} ({top_category.score:.2f})"

    return {"result": result}






