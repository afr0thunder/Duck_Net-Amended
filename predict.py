import torch
import numpy as np
from PIL import Image, ImageOps

from model import DuckNet
from utils.config import Configs


def load_model(config:Configs, model_path:str):
    model = DuckNet(config.input_channels, config.num_classes, config.num_filters)
    model.load_state_dict(torch.load(model_path))
    return model


def predict(model:DuckNet, image_path:str, device:torch.device):
    image = ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')
    image = np.array(image.resize((512, 512)))
    image = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
    print(output.min(), output.max())
    output_image = torch.clamp(output, 0, 1).squeeze().cpu().numpy()
    print(output_image.shape)
    return output_image


def main(config:Configs, model_path:str, image_path:str, output_path:str):
    device = torch.device(f'cuda:{config.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = load_model(config, model_path)
    model.to(device)
    print(f'Model loaded from {model_path}')

    output_image = predict(model, image_path, device)
    output_image = Image.fromarray((output_image * 255).astype(np.uint8))
    output_image.save(output_path)
    print(f'Prediction saved at {output_path}')


if __name__ == '__main__':
    config = Configs()
    model_path = 'checkpoints/best_model.pt'
    image_path = 'sample.jpg'
    output_path = 'output.jpg'
    main(config, model_path, image_path, output_path)