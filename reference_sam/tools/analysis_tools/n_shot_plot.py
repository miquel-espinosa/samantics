import re
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse_coco_results(text):
    results = {}
    pattern = r'[AP|AR]\) @\[ IoU=[0\.50|0\.75].*?\| area=.*? \| maxDets=\d+ \] = (\d+\.\d+)'
    matches = re.findall(pattern, text)
    return matches


def main():

    # folder with the list of files to be analyzed
    folder = "n_shot_large"

    keys = [
        "AP @[ IoU=0.50:0.95 | area= all | maxDets=100 ]",
        "AP @[ IoU=0.50 | area= all | maxDets=1000 ]",
        "AP @[ IoU=0.75 | area= all | maxDets=1000 ]",
        "AP @[ IoU=0.50:0.95 | area= small | maxDets=1000 ]",
        "AP @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ]",
        "AP @[ IoU=0.50:0.95 | area= large | maxDets=1000 ]",
        "AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
        "AR @[ IoU=0.50:0.95 | area=   all | maxDets=300 ]",
        "AR @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ]",
        "AR @[ IoU=0.50:0.95 | area= small | maxDets=1000 ]",
        "AR @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ]",
        "AR @[ IoU=0.50:0.95 | area= large | maxDets=1000 ]",
    ]

    results = {}

    # Sort files by n_shot, 1, 5, 10, 20, 50, 100
    files = os.listdir(folder)
    files = [f for f in files if f.endswith(".out")]
    files.sort(key=lambda x: int(re.search(r"(\d+)-shot", x).group(1)))

    # Take only .out files
    for filename in files:
        n_shot = re.search(r"(\d+)-shot", filename).group(1)
        with open(os.path.join(folder, filename), "r") as file:
            text = file.read()

        # Parse COCO results
        matches = parse_coco_results(text)

        for i, key in enumerate(keys):
            if key not in results:
                results[key] = {}
            results[key][n_shot] = float(matches[i])


    # Create subplots
    fig, axs = plt.subplots(4, 3, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    axs = axs.ravel()

    # Plot data for each key
    for i, key in enumerate(keys):
        ax = axs[i]
        ax.set_title(key)
        ax.set_xlabel('n_shot')
        ax.set_ylabel('Value')
        n_shot_values = list(results[key].keys())
        metric_values = list(results[key].values())
        ax.plot(n_shot_values, metric_values, color='black', zorder=1)
        ax.scatter(n_shot_values, metric_values, marker='o', color='red', zorder=2)

    plt.tight_layout()
    plt.savefig(f"{folder}/n_shot_large.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
