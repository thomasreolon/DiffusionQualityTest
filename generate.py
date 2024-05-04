import json
from tqdm import tqdm
import torch
from diffusers import DiffusionPipeline
import os
import gc

MODELS = [
    'stabilityai/stable-diffusion-xl-base-1.0', # needs 10GB GPU RAM
    'RunDiffusion/Juggernaut-X-v10', # needs 10GB GPU RAM
    'runwayml/stable-diffusion-v1-5',
]
IMG_FOLDER = 'output_folder'
os.makedirs(IMG_FOLDER, exist_ok=True)

# Load a model
def load_pipeline(model):
    gc.collect()
    torch.cuda.empty_cache()
    pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16).to("cuda")
    pipe.safety_checker = None
    def generate_img(prompt, out_file):
        gc.collect()
        torch.cuda.empty_cache() 
        image = pipe(prompt).images[0]
        image.save(out_file)
    return generate_img

# Loop Generation
prompts = json.load(open('prompts.json'))
looper = tqdm(range(len(MODELS)*26*40))
for model in MODELS:
    model_name = model.split('/')[1]

    # Load model in GPU if not all images generated
    ended = [x for x in os.listdir('output_folder') if (model_name in x) and ('25' in x)]
    if len(ended) < 40 :
        print('---> LOADING', model)
        generate_img = load_pipeline(model)

    # Generate image if not already present
    for i in range(26):
        for kind in prompts:
            for img in prompts[kind]:
                looper.update(1)
                save_path = f'{IMG_FOLDER}/{model_name}|{kind}|{img}|{i:02d}.png'
                if not os.path.isfile(save_path):
                    generate_img(prompts[kind][img], save_path)

with open('ended.txt', 'w') as f:
    f.write('ok')