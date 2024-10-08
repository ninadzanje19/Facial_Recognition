from fastapi import FastAPI
from variables import *
from Functions import extract_facial_embeddings
from pydantic import BaseModel
from detect_face import detect_face
app = FastAPI()

@app.get("/get_embeddings")
async def get_embeddings():
    embeddings = extract_facial_embeddings(r'G:\Projects\Facial_Recognition\Facial_Recognition\data\test\Zaheer_Khan_Test.jpg')
    return {"Facial Embeddings: " : embeddings}


class facial_metadata(BaseModel):
    url: str

@app.post("/get_facial_embeddings")
async def get_embeddings(fc: facial_metadata):
    embeddings = extract_facial_embeddings(fr"{fc.url}")
    return {"Facial Embeddings: " : embeddings}

class face_data(BaseModel):
    facial_train_data: str
    facial_test_file: str
@app.post("/detect_face")
async def face_detection(fd:face_data):
    detection = detect_face(fr"{fd.facial_train_data}", fr"{fd.facial_test_file}")
    return {detection}