import os

for filename in os.listdir("./CelebA-HQ-img/test"):
    os.replace(f"./CelebAMask-HQ-mask-processed/{filename.split('.')[0]}.png", f"./CelebAMask-HQ-mask-processed/test/{filename.split('.')[0]}.png")
for filename in os.listdir("./CelebA-HQ-img/train"):
    os.replace(f"./CelebAMask-HQ-mask-processed/{filename.split('.')[0]}.png", f"./CelebAMask-HQ-mask-processed/train/{filename.split('.')[0]}.png")
for filename in os.listdir("./CelebA-HQ-img/val"):
    os.replace(f"./CelebAMask-HQ-mask-processed/{filename.split('.')[0]}.png", f"./CelebAMask-HQ-mask-processed/val/{filename.split('.')[0]}.png")
