import numpy as np
from data.dataloader_utils import get_forget_retain_loaders

if __name__ == '__main__':
    dummy_forget = np.random.choice(50000, size=100, replace=False)
    forget_loader, retain_loader = get_forget_retain_loaders(dummy_forget)

    f_imgs, f_labels = next(iter(forget_loader))
    r_imgs, r_labels = next(iter(retain_loader))

    print(f"Forget batch shape: {f_imgs.shape}")
    print(f"Retain batch shape: {r_imgs.shape}")
    print("DataLoader utility works correctly")
