import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

filename = "couch2.jpeg"

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cpu")
midas.to(device)
midas.eval()

transform = midas_transforms.small_transform

img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)

with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()
print(output)