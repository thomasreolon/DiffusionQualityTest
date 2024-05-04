import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns

import numpy as np
import json
import pandas as pd

from scoring import RESULTS_FILE

def make_plot(x,y, color="plasma", title="", y_max=None, y_min=None, x_axis=None):
    x = np.array(x)
    y = np.array(y)
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points together easily to get the segments.
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create the LineCollection object, specifying the linewidth and the color map
    lc = LineCollection(segments, cmap=color, norm=plt.Normalize(0, 10))
    lc.set_array(x)
    lc.set_linewidth(5)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    y_max = y_max or y.max()
    y_min = y_min or y.min()
    ax.set_ylim(y_min, y_max)
    ax.set_title(title)

    # Adding data labels
    l = len(x)
    for i, (xi, yi) in enumerate(zip(x, y)):
        if i<2 or i==7 or i==l//3 or i==l//2 or i==l*3//4 or i==l-1: 
            ax.text(xi, yi, f'{yi:.2f}', color='grey', fontsize=8)

    if x_axis:  # only set the x-axis label if x_ax is provided
        ax.set_xlabel(x_axis)
    plt.colorbar(lc, ax=ax, label='Score by CLIP-IQA')
    plt.show()


def make_distrib(scores):
    # Set the style of seaborn plot
    sns.set_theme(style="whitegrid")

    # Create a histogram with a kernel density estimate
    sns.histplot(scores, kde=True, color='blue', bins=10)  # You can adjust the number of bins

    # Add titles and labels
    plt.title('Distribution of Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()

with open(RESULTS_FILE) as f_in:
    data = json.load(f_in)


df = []
for fname, score in data.items():
    model, categ, image, id = fname[:-4].split('|')
    df.append({
        'model':model,
        'category':categ,
        'image':image,
        'id':int(id),
        'score':float(score),
        'original':fname,
    })

df = pd.DataFrame(df)

make_distrib(df.score.tolist())

# Chart
max_across_models = df.groupby('model').id.max().min()

for cat in df.category.unique():
    df_c = df[ df.category==cat ]
    df_c = df_c[df_c.id <= max_across_models]
    chart_y = []
    for i in range(max_across_models):
        x = []
        for j in range(i+1):
            sample = df_c.sample(400)
            sample = sample.score.tolist()
            x.append(sample)
        x = np.array(x)  # ix400
        expected_score = x.max(axis=0).mean()
        chart_y.append(expected_score)
    chart_x = list(range(len(chart_y)))
    print(cat, chart_y[0], chart_y[-1])
    make_plot(chart_x, chart_y, 'copper_r', title=cat.title(), y_max=0.9, y_min=0.6)


for model in df.model.unique():
    df_c = df[ df.model==model ]
    chart_y = []
    for i in range(max_across_models):
        x = []
        for j in range(i+1):
            sample = df_c.sample(400)
            sample = sample.score.tolist()
            x.append(sample)
        x = np.array(x)  # ix400
        expected_score = x.max(axis=0).mean()
        chart_y.append(expected_score)
    chart_x = list(range(len(chart_y)))
    print(model, chart_y[0], chart_y[-1])
    make_plot(chart_x, chart_y, 'copper', title=model.title(), y_max=0.9, y_min=0.6)
