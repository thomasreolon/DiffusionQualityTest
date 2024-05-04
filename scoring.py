import os, json
import pyiqa
import torch

from generate import IMG_FOLDER

RESULTS_FILE = f'evaluation.json'


if not os.path.isfile(RESULTS_FILE):
    # Load NN
    metric = pyiqa.create_metric('clipiqa', device=torch.device("cuda"))

    # Evaluate each image
    score_images = {}
    for f in os.listdir(IMG_FOLDER):
        img_path = os.path.join(IMG_FOLDER, f)
        score = metric(img_path)
        score_images[f] = score.mean().item()

    # Save results
    with open(RESULTS_FILE, 'w') as file:
        json.dump(score_images, file)

