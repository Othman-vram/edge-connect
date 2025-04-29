from PIL import Image, ImageOps, ImageFilter
import os
from collections import defaultdict

from multiprocessing import Pool
import time

# === Settings ===
input_folder = 'CelebAMask-HQ-mask-anno'  # folder where your images are stored
output_folder = 'CelebAMask-HQ-mask-processed'  # folder to save assembled images
layers_names = [
    "neck",
    "cloth",
    "skin",
    "hair",
    "u_lip",
    "l_lip",
    "l_brow",
    "r_brow",
    "l_eye",
    "r_eye",
    "l_ear",
    "r_ear",
    "nose",
    "mouth",
    "ear_r",
    "ear_g",
    "eye_g",
    "hat"
]

layer_colors = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (125, 255, 0),
    (125, 0, 255),
    (255, 125, 0),
    (255, 0, 125),
    (0, 255, 125),
    (0, 125, 255),
    (125, 125, 0),
    (125, 0, 125),
    (0, 125, 125),
    (60, 255, 0),
    (60, 0, 255),
    (0, 60, 255),
]

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# === Load and group images ===
images_by_group = defaultdict(list)

def process(tuple):
    group_key = tuple[0]
    images = tuple[1]
    layers = []

    for i in range(len(layers_names)):
        img_path = os.path.join(input_folder, group_key + '_' + layers_names[i] + '.png')
        try:
            img = Image.open(img_path).convert('RGBA')

            # Convert black to transparent
            datas = img.getdata()
            new_data = []
            for item in datas:
                if item[0] == 0 and item[1] == 0 and item[2] == 0:
                    # Fully transparent
                    new_data.append((0, 0, 0, 0))
                else:
                    new_data.append(item)
            img.putdata(new_data)

            # Colorize layer
            color = layer_colors[i]

            # Tint the non-transparent parts
            r, g, b, a = img.split()
            tint_r = ImageOps.colorize(r, black="black", white=f"rgb{color}").split()[0]
            tint_g = ImageOps.colorize(g, black="black", white=f"rgb{color}").split()[1]
            tint_b = ImageOps.colorize(b, black="black", white=f"rgb{color}").split()[2]
            colored_img = Image.merge('RGBA', (tint_r, tint_g, tint_b, a))

            layers.append(colored_img)

        except FileNotFoundError:
            pass

    base = Image.new('RGBA', layers[0].size, (0, 0, 0, 0))

    for layer in layers:
        base = Image.alpha_composite(base, layer)

    # Save result
    output_path = os.path.join(output_folder, f'{str(int(group_key))}.png')
    base = base.convert("L")
    base = base.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8,
                                           -1, -1, -1, -1), 1, 0))
    base = base.point(lambda p: 255 if p > 50 else 0)
    base.save(output_path)

    print(f'Saved {output_path}')

# Collect images
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        group_key = filename[:5]
        rest_key = filename[5:]
        images_by_group[group_key].append((rest_key, filename))

def execute_task_parallel(tuple):
    with Pool(80) as p:
        return p.map(process, tuple)

results = execute_task_parallel(images_by_group.items())
# results = process(('00000', images_by_group['00000']))
