import torch
import json
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torchvision.io import read_image
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image

def load_class_names(json_path='../data/imagenet_class_index.json'):
    """Laddar ImageNet-klassnamn från JSON-fil."""
    try:
        with open (json_path, 'r') as f:
            class_idx = json.load(f)
        return {int(k): v[1] for k, v in class_idx.items()}
    except FileNotFoundError:
        print(f'Fel: Hittade inte {json_path}')
        return None

def get_model():
    """Laddar en förtränad ResNet18-modell."""
    model = models.resnet18(weights='IMAGENET1K_V1').eval()
    return model

def preprocess_image(img_path):
    """Läser in och förbreder bilden för modellen."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    ])
    raw_img = read_image(img_path)
    input_tensor = transform(to_pil_image(raw_img)).unsqueeze(0)
    return input_tensor, raw_img

def generate_cam(model, input_tensor, raw_img, target_class_idx):
    """Genererar en värmekarta för en specifik klass."""
    cam_extractor =GradCAM(model)

    out = model(input_tensor)

    activation_map = cam_extractor(target_class_idx, out)

    result = overlay_mask(to_pil_image(raw_img).resize((224, 224)),
                          to_pil_image(activation_map[0], mode='F'),
                          alpha=0.5)
    
    cam_extractor.remove_hooks()
    return result, out

def get_top_predictions(output, class_names, k=5):
    """Hämtar de k högsta gissningarna med namn från JSON-filen."""
    probs = torch.nn.functional.softmax(output, dim=1)
    top_probs, top_idxs = probs.topk(k)

    results = []
    for i in range(k):
        idx = top_idxs[0][i].item()
        prob = top_probs[0][i].item()
        name = class_names[idx] if class_names else f"ID {idx}"
        results.append((name, prob))
    return results

def show_cam(original_img, cam_img, title=''):
    """Visar originalbild och CAM-värmekarta sida vid sida"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(to_pil_image(original_img).resize((224, 224)))
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(cam_img)
    axes[1].set_title(f'CAM: {title}')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    
    