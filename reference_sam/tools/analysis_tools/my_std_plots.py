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
    folder = "std_coco_results"

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

    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), "r") as file:
            text = file.read()

        # Parse COCO results
        matches = parse_coco_results(text)

        for i, key in enumerate(keys):
            if key not in results:
                results[key] = []
            results[key].append(float(matches[i]))

    # Create a pandas DataFrame
    df = pd.DataFrame(results, columns=keys)

    # Plotting
    plt.figure(figsize=(12, 12))
    sns.violinplot(data=df, inner="points")
    plt.xticks(rotation=90)
    plt.title(f"Variance of COCO Results for {len(df)} runs")
    plt.ylabel("Value")
    plt.xlabel("Keys")
    plt.tight_layout()
    plt.savefig(f"{folder}/coco_results.png", dpi=300)


if __name__ == "__main__":
    main()
