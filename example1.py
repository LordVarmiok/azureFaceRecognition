import os
import io
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
import requests
from PIL import Image, ImageDraw

import vault
import image_urls

"""
Przykład 1: Wykrywanie twarzy na obrazie
    - rysowanie kwadratu dookoła wykrytej twarzy
"""

API_KEY = vault.key1
ENDPOINT = vault.endpoint

face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))

target_image_url = image_urls.urls['rhcp']  # change image here
image_name = os.path.basename(target_image_url)

response_detected_faces = face_client.face.detect_with_url(
    target_image_url,
    detection_model='detection_03',
    recognition_model='recognition_04'
)

# print(response_detected_faces)

if not response_detected_faces:
    raise Exception('No face detected')
else:
    print('Number of people detected: ', len(response_detected_faces))
    person1 = response_detected_faces[0]

    response_image = requests.get(target_image_url)
    img = Image.open(io.BytesIO(response_image.content))
    drawing = ImageDraw.Draw(img)

    # draw a rectangle around each of the detected faces in the image
    for face in response_detected_faces:
        rect = face.face_rectangle
        left = rect.left
        top = rect.top
        right = rect.width + left
        bottom = rect.height + top
        drawing.rectangle(((left, top), (right, bottom)), outline='green', width=5)
    img.show()
