from pathlib import Path
import click
from json import loads
import numpy as np

ALLOW_ALL = True
ALLOWED_CATEGORIES = ["bicycling", "running", "sports", "transportation", "walking"]


@click.command
def main():
    p = (Path(__file__) / "../../pose_data/annotations.json").resolve()

    # Load the annotation data
    with open(p) as fr:
        json_str = fr.read()
    annotations = loads(json_str)

    # Parse out the categories we care about
    valid_imgs = []
    for idx, act in enumerate(annotations["act"]):
        if ALLOW_ALL or act["cat_name"] in ALLOWED_CATEGORIES:
            valid_imgs.append(idx)

    # Split the data into train and test
    TEST_PERCENT = 0.1
    n_test = int(np.floor(len(valid_imgs) * TEST_PERCENT))

    np.random.shuffle(valid_imgs)  # TODO: Fix random seed for reproducibility
    test, train = valid_imgs[:n_test], valid_imgs[n_test:]

    print(
        f"Total dataset size: {len(valid_imgs)}, training with: {len(train)}, testing with: {len(test)}"
    )


if __name__ == "__main__":
    main()
