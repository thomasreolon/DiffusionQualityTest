# Diffusion Experiment

Code that generates a lot of images with diffusion models, evaluates them, plots some statistics.

## Getting Started

You can replicate the results in the paper: [Exploring Generative Models for High-Quality Image Synthesis](link)

To run the code you need a GPU with 10+ GB of RAM.

Then you need to install: `torch`, `diffusers`, `pyiqa` and (`tqdm`, `seaborn`, `pandas`).

```bash
# once you have the environment set up, you can run
bash pipeline.sh
```

## Conclusion

Tom improve the final quality of generated images you can:

1. Generate _N_ images from the same prompt
2. Select the image with the best score (eg. using pyiqa metrics)

In the paper we discover that for generic image generation:
- _N_ should be < 20
- Most results are achieved with an _N_ around 5
- This method improves the expected image quality of 15-20% (measured by CLIP-IQA)


