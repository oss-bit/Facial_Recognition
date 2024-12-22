
import torchvision
import cv2
from chromadb import PersistentClient
from utils import get_dataloader,transform
from uuid import uuid5

proximity_threshold = 80
similarity_threshold = 0.3
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights)


face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

vector_db = PersistentClient(path='embeddings')
vector_db.create_collection(
    'ficial_recognition',
    ndim=2048,
    metric='cossine'
)

def load_data_dir(dir_path):
    fl_ids = []
    data_loader = get_dataloader(dir_path)
    for batch in data_loader:
        features = model.features(batch)
        features = [ feature.squeeze() for feature in features]
        ids = [uuid5 for _ in range(len(features))]
        vector_db.add(
            embeddings=features,
            ids=ids
        )
        fl_ids.append(*ids)


            
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break
    faces = face_detector.detectMultiScale(img,1.3,4)
    l_face = max(faces, key=lambda x: x[2] * x[3]) #return the max based on the area of the faces
    if l_face[2] * l_face[3] < proximity_threshold:
        continue
    d_img = img[int(l_face[1]):int(l_face[1]+l_face[3]), int(l_face[0]):int(l_face[0]+l_face[2])] #croping the face from the image
    d_img = transform(d_img).unsqueeze(0)
    feature = model.features(d_img).squeeze()
    query = vector_db.query(
                        query_emdeddings=feature,
                        n_results=2
                        )
    for distance in enumerate(query['distance']):
        if distance < similarity_threshold:
            cv2.rectangle(img,(l_face[0],l_face[1]),(l_face[0]+l_face[2],l_face[1]+l_face[3]),(255,0,0),5) #draw rectangle to main image









