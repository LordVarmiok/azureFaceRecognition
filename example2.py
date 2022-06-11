from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image, ImageDraw

import vault

"""
Przykład 2: Wykrywanie elementów twarzy na obrazie
    - rysowanie kwadratu dookoła wykrytego elementu twarzy
"""

API_KEY = vault.key1
ENDPOINT = vault.endpoint

face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))

target_image_path = 'images/inauguration.jpg'  # change image here
image_file = open(target_image_path, 'rb')

response_detected_faces = face_client.face.detect_with_stream(
    image=image_file,
    detection_model='detection_03',
    recognition_model='recognition_04',
    return_face_landmarks=True
)

print(response_detected_faces)

if not response_detected_faces:
    raise Exception('No face detected')
else:
    print('Number of people detected: ', len(response_detected_faces))
    person1 = response_detected_faces[0]

    print(vars(person1.face_landmarks).keys())

    img = Image.open(image_file)
    drawing = ImageDraw.Draw(img)

    # draw a rectangle around each of the detected faces in the image
    for face in response_detected_faces:
        rect = face.face_rectangle
        left = rect.left
        top = rect.top
        right = rect.width + left
        bottom = rect.height + top
        drawing.rectangle(((left, top), (right, bottom)), outline='green', width=5)

        # mark the noise tip
        x = face.face_landmarks.nose_tip.x
        y = face.face_landmarks.nose_tip.y
        drawing.rectangle(((x, y), (x, y)), outline='white', width=7)

        # draw the bounding box around the mouth
        mouth_left = face.face_landmarks.mouth_left.x, face.face_landmarks.mouth_left.y
        mouth_right = face.face_landmarks.mouth_right.x, face.face_landmarks.mouth_right.y
        lip_bottom = face.face_landmarks.under_lip_bottom.x, face.face_landmarks.under_lip_bottom.y
        drawing.rectangle((mouth_left, (mouth_right[0], lip_bottom[1])), outline='yellow', width=2)

    img.show()
