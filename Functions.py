"Functions used accross the code are written here"


from pickletools import stringnl
from PIL import Image
import pathlib as Path
import cv2
import numpy as np
from PIL import Image
from imgbeddings import *
from variables import test_data_file, train_data_dir

#converts any image to jpg, with a small edit this function can be made to convert to many other image file types
def convert_img_to_jpg(input_image, output_address):
    with Image.open(input_image) as img:
        img = img.convert("RGB")
        img.save(output_address, "JPEG")

#converts the address string given to a format accepted by the SwaggerAPI
def get_SwaggerAPI_str(string):
    string = str(string).split("\\")
    SwaggerAPI_str = ""
    for i in string:
        if i == string[-1]:
            SwaggerAPI_str = SwaggerAPI_str + i
        else:
            SwaggerAPI_str = SwaggerAPI_str + i + "/"

    return SwaggerAPI_str

#get the embeddings of the image address given
def extract_facial_embeddings(image_address):
    file_name = image_address
    img = Image.open(file_name)
    ibed = imgbeddings()
    embedding = ibed.to_embeddings(img)
    embedding = embedding[0]
    return embedding.tolist()

