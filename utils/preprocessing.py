import torchvision.transforms as T
from PIL import Image
def preprocess_crop(frame, box):
    x1, y1, x2, y2 = box
    crop = Image.fromarray(frame[y1:y2, x1:x2])
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(crop)