from PIL import Image
import pathlib as Path
import cv2
import numpy as np
from PIL import Image
from imgbeddings import *

from variables import test_data_file


def convert_img_to_jpg(input, output):
    with Image.open(input) as img:
        img = img.convert("RGB")
        img.save(output, "JPEG")



def extract_facial_embeddings(image_address):
    # loading the face image path into file_name variable
    file_name = image_address  # replace <INSERT YOUR FACE FILE NAME> with the path to your image
    # opening the image
    img = Image.open(file_name)
    # loading the `imgbeddings`
    ibed = imgbeddings()
    # calculating the embeddings
    embedding = ibed.to_embeddings(img)
    embedding = embedding[0]
    return embedding.tolist()

#print(extract_facial_embeddings(test_data_file))