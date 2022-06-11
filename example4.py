from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image, ImageDraw, ImageFont

import vault

"""
Przykład 4: Sprawdzanie czy wykryta twarz pojawia się na innych zdjęciach
    - rysowanie kwadratu dookoła wykrytego elementu twarzy
    - sprawdzanie czy wykryta twarz pojawia się na innych zdjęciach
"""

API_KEY = vault.key1
ENDPOINT = vault.endpoint

face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))

target_image_path = 'images/joe2.jpg'  # change image here
image_file = open(target_image_path, 'rb')

response_detected_faces = face_client.face.detect_with_stream(
    image=image_file,
    detection_model='detection_03',
    recognition_model='recognition_04',
)
face_ids = face_ids = [face.face_id for face in response_detected_faces]

target_image_path2 = './images/joe1.jpg' # change second image here
img_source2 = open(target_image_path2, 'rb')
response_face_source = face_client.face.detect_with_stream(
    image=img_source2,
    detection_model='detection_03',
    recognition_model='recognition_04'
)
face_id_source = response_face_source[0].face_id

matched_faces = face_client.face.find_similar(
    face_id=face_id_source,
    face_ids=face_ids
)

if not response_detected_faces or not response_face_source:
    raise Exception('No face detected in both images')
else:
    img = Image.open(image_file)
    drawing = ImageDraw.Draw(img)
    font = ImageFont.truetype('C:/Windows/Fonts/Calibri.ttf', 25)

    for matched_face in matched_faces:
        for face in response_detected_faces:
            if face.face_id == matched_face.face_id:
                rect = face.face_rectangle
                left = rect.left
                top = rect.top
                right = rect.width + left
                bottom = rect.height + top
                drawing.rectangle(((left, top), (right, bottom)), outline='green', width=5)
img.show()

face_verified = face_client.face.verify_face_to_face(
    # face_id1=matched_faces[0].face_id,
    face_id1=face_ids[0],
    face_id2=face_id_source
)
print('Are these the same faces? Answer: ', face_verified.is_identical)