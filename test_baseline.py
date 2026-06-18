import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

# CIFAR-100 class names
testset = torchvision.datasets.CIFAR100(
    root='./data',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
)

classes = testset.classes

# Build model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 100)

# Load checkpoint
checkpoint = torch.load("models/baseline.pt", map_location="cpu")

# Try both checkpoint formats
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()
import random

for n in range(5):

    # Pick random test image
    idx = random.randint(0, len(testset) - 1)
    img, label = testset[idx]

    # Run model
    with torch.no_grad():
        output = model(img.unsqueeze(0))

        probs = F.softmax(output, dim=1)

        top5_probs, top5_indices = torch.topk(probs, 5)

    print("\n" + "=" * 50)
    print(f"Image {n+1}")
    print(f"TRUE LABEL: {classes[label]}")
    print("\nTop 5 Predictions:")

    for prob, pred_idx in zip(top5_probs[0], top5_indices[0]):
        print(
            f"{classes[pred_idx]:15s} "
            f"{prob.item()*100:.2f}%"
        )

    # Un-normalize image for display
    display_img = img.permute(1, 2, 0)

    mean = torch.tensor([0.5071, 0.4867, 0.4408])
    std = torch.tensor([0.2675, 0.2565, 0.2761])

    display_img = display_img * std + mean
    display_img = display_img.clamp(0, 1)

    plt.figure(figsize=(4, 4))
    plt.imshow(display_img)
    plt.title(
        f"True: {classes[label]}\n"
        f"Pred: {classes[top5_indices[0][0]]}"
    )
    plt.axis("off")
    plt.show()
