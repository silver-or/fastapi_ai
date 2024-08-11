import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from PIL import Image
import io
import numpy as np

from fastapi import FastAPI, File, UploadFile

# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='models\efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

app = FastAPI()


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()

    # STEP 3: Load the input image. 
    pil_img = Image.open(io.BytesIO(contents))
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

    # STEP 4: Detect objects in the input image.
    detection_result = detector.detect(image)  

    # STEP 5. 찾은 객체의 종류와 종류 개수를 출력하시오.
    result_dict = {}
    for detection in detection_result.detections:
        category = detection.categories[0].category_name
        if category not in result_dict:
            result_dict[category] = 1
        else:
            result_dict[category] += 1

    return {"result": result_dict}