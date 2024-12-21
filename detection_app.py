
import torchvision
import cv2
from chromadb import PersistentClient
from utils import get_dataloader




weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights)


face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# vector_db = PersistentClient(path='/embeddings')
# vector_db.create_collection(
#     'ficial_recognition',
#     ndim=2048,
#     metric='cossine'
# )

def load_data_dir(dir_path):
    data_loader = get_dataloader(dir_path)
    for batch in data_loader:
        print(batch)
        break

            
if __name__ == '__main__':

    load_data_dir('backgrounds/')







