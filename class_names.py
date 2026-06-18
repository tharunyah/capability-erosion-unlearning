import numpy as np
import torchvision

acc = np.load("data/baseline_per_class_acc.npy")

dataset = torchvision.datasets.CIFAR100(
    root="./data",
    train=False,
    download=True
)

best = np.argsort(acc)[-10:]
worst = np.argsort(acc)[:10]

print("\nBEST CLASSES")
for idx in reversed(best):
    print(dataset.classes[idx], f"{acc[idx]*100:.1f}%")

print("\nWORST CLASSES")
for idx in worst:
    print(dataset.classes[idx], f"{acc[idx]*100:.1f}%")
target = "turtle"

for i in range(len(testset)):
    img, label = testset[i]

    if classes[label] == target:
        ...
