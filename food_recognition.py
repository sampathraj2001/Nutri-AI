from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import torch

# Loading the model from Hugging Face
model = ViTForImageClassification.from_pretrained("rajistics/finetuned-indian-food")
feature_extractor = ViTFeatureExtractor.from_pretrained("rajistics/finetuned-indian-food")

# preprocessing the image
def preprocess_image(image_path):
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs


def predict_food(image_path):
    inputs = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(**inputs)
    
    #predicted label
    predicted_class = outputs.logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class]
    return ''.join([word.capitalize() for word in predicted_label.split('_')])


if __name__ == "__main__":
    image_path = "C:/Users/hp/Downloads/Pizza.JPG"  
    predicted_food = predict_food(image_path)
    print(f"Predicted Indian Food: {predicted_food}")