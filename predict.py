import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model_v2 import get_rop_model
import sys

def predict_single_image(img_path, model_path='best_rop_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = get_rop_model(num_classes=2, pretrained=False)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(device)
    model.eval()

    # 2. Preprocess Image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 3. Inference
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        
        prob_rop = probs[0][1].item()
        prob_normal = probs[0][0].item()
        
        _, predicted = output.max(1)
        class_name = "ROP" if predicted.item() == 1 else "Normal"
        confidence = prob_rop if predicted.item() == 1 else prob_normal

    # 4. Visualize
    plt.imshow(image)
    plt.title(f"Prediction: {class_name} ({confidence*100:.2f}% confidence)")
    plt.axis('off')
    plt.show()

    print(f"Image: {img_path}")
    print(f"Result: {class_name} | Probability of ROP: {prob_rop:.4f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_single_image(sys.argv[1])
    else:
        print("Usage: python predict.py path/to/image.jpg")
