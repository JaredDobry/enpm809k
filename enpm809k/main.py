from pathlib import Path
import click
from json import loads
import numpy as np
from glob import glob
from wget import download
import tarfile

ALLOW_ALL = True
ALLOWED_CATEGORIES = ["bicycling", "running", "sports", "transportation", "walking"]
IMAGE_URL = (
    "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz"
)


def get_dataset():
    # Check if we already have the dataset downloaded
    if len(glob("./pose_data/images/*.jpg")) == 0:
        print("Dataset not found locally, pulling it...")
        pose_data = Path(__file__).parent.parent / "pose_data"
        images = pose_data / "images"
        if not pose_data.exists():
            pose_data.mkdir()
            if not images.exists():
                images.mkdir()
        temp_path = (
            Path(
                __file__,
            ).parent.parent
            / "temp.tar.gz"
        ).as_posix()
        download(IMAGE_URL, temp_path)
        tar = tarfile.open(temp_path)
        tar.extractall(images)


@click.command
def main():
    # Ensure dataset is available
    get_dataset()

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
