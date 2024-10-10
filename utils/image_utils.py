import torch
import numpy as np
from torchvision import transforms
import cv2

def PIL2Torch(pil_image):
    torch_image = torch.from_numpy(np.array(pil_image)) / 255.0
    if len(torch_image.shape) == 3:
        return torch_image.permute(2, 0, 1)
    else:
        return torch_image.unsqueeze(dim=-1).permute(2, 0, 1)

def Torch2PIL(torch_image):
    pil_image = (torch_image.cpu().detach().permute(1, 2, 0).numpy() * 255).copy()
    return pil_image.astype(np.uint8)

def save_img(image, filename, image_type="torch"):
    if image_type.lower() == "torch":
        image = Torch2PIL(image)
    if "." not in filename:
        filename += ".png"
    cv2.imwrite(filename, image[:, :, ::-1])

def rgb2loftrgray(img):
    resizer = transforms.Resize([480,640])
    gray=transforms.functional.rgb_to_grayscale(img)
    img11 = resizer(gray)
    img11 = img11[None].cuda()
    return img11