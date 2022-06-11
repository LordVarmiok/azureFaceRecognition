from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image, ImageDraw, ImageFont

import vault

"""
Przykład 3: Wykrywanie emocji i wieku osoby, której twarz została wykryta
    - rysowanie kwadratu dookoła wykrytej głowy
    - wypisywanie wieku oraz wykrytych emocji w wartościach procentowych
"""

API_KEY = vault.key1
ENDPOINT = vault.endpoint

face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))

target_image_path = 'images/joe_and_jill.jpg'  # change image here
image_file = open(target_image_path, 'rb')

response_detected_faces = face_client.face.detect_with_stream(
    image=image_file,
    detection_model='detection_01',
    recognition_model='recognition_04',
    return_face_attributes=['age', 'emotion']
)

print(response_detected_faces)

if not response_detected_faces:
    raise Exception('No face detected')
else:
    print('Number of people detected: ', len(response_detected_faces))
    person1 = response_detected_faces[0]

    img = Image.open(image_file)
    drawing = ImageDraw.Draw(img)
    font = ImageFont.truetype('C:/Windows/Fonts/Calibri.ttf', 25)

    # draw a rectangle around each of the detected faces in the image
    for face in response_detected_faces:
        rect = face.face_rectangle
        left = rect.left
        top = rect.top
        right = rect.width + left
        bottom = rect.height + top
        drawing.rectangle(((left, top), (right, bottom)), outline='green', width=5)

        age = face.face_attributes.age
        emotion = face.face_attributes.emotion
        neutral = '{0:.0f}%'.format(emotion.neutral * 100)
        happiness = '{0:.0f}%'.format(emotion.happiness * 100)
        anger = '{0:.0f}%'.format(emotion.anger * 100)
        sandness = '{0:.0f}%'.format(emotion.sadness * 100)
        print(vars(emotion)) # printing emotion values - 0.0 is 0%, 1.0 is 100%

        drawing.text((right + 4, top), 'Age: ' + str(int(age)), fill=(255, 255, 255), font=font)
        drawing.text((right + 4, top+35), 'Neutral: ' + neutral, fill=(255, 255, 255), font=font)
        drawing.text((right + 4, top+70), 'Happy: ' + happiness, fill=(255, 255, 255), font=font)
        drawing.text((right + 4, top+105), 'Sad: ' + sandness, fill=(255, 255, 255), font=font)
        drawing.text((right + 4, top+140), 'Angry: ' + anger, fill=(255, 255, 255), font=font)

    img.show()
