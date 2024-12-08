import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

filename = "couch2.jpeg"
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
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
# print(np.max(output))
# print(output)
output = output / 1200
# print(output)
# h, w = output.shape
# vis2 = cv2.CreateMat(h, w, cv2.CV_32FC3)
# img = cv2.CvtColor(output, cv2.CV_GRAY2BGR)
cv2.imshow("WindowNameHere", output)
# cv2.waitKey(0)

# [107.71465301513672, 468.7779235839844, 744.2359008789062, 853.68994140625]
# rect = patches.Rectangle((107.71465301513672, 468.7779235839844), 744.2359008789062 - 107.71465301513672, 853.68994140625 - 468.7779235839844, linewidth=1, edgecolor='r', facecolor='none')
fig, ax = plt.subplots()
ax.imshow(output)
# ax.add_patch(rect)
plt.show()

