from fastapi import FastAPI
from variables import *
from Functions import extract_facial_embeddings
from pydantic import BaseModel
from detect_face import detect_face
app = FastAPI()

#home route
@app.get("/")
async def get_embeddings():
    message = "Welcome to Facial Recognition"
    return {message}


#post route to get facial embeddings of an image
class facial_metadata(BaseModel):
    url: str
@app.post("/get_facial_embeddings")
async def get_embeddings(fc: facial_metadata):
    embeddings = extract_facial_embeddings(fr"{fc.url}")
    return {"Facial Embeddings: " : embeddings}


#post route to detect face from a given pool of faces
class face_data(BaseModel):
    facial_train_data: str
    facial_test_file: str
@app.post("/detect_face")
async def face_detection(fd:face_data):
    detection = detect_face(fr"{fd.facial_train_data}", fr"{fd.facial_test_file}")
    return {detection}