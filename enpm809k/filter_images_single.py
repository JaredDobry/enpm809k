# Load the annotation data
from json import load, dump
from pathlib import Path

data_dir = "E:/Code/enpm809k/pose_data"
p = (Path(data_dir) / "annotations.json").resolve()
annotations = load(p.open())

# Load annotated images in the following categories
ALLOW_ALL = False
ALLOWED_CATEGORIES = ["bicycling", "running", "walking"]
ALLOWED_JOINTS = [2, 3, 9, 12, 13] # r hip, l hip, head, r shoulder, l shoulder


valid_imgs = []
for i in range(len(annotations["act"])):
    if ALLOW_ALL or annotations["act"][i]["cat_name"] in ALLOWED_CATEGORIES:
        valid_imgs.append(i)
print(f"Indexed {len(valid_imgs)} images with valid categories")

# Parse it to only images with a single person visible
annorect_imgs = []

for i in range(len(valid_imgs)):
    i_img = valid_imgs[i]
    annolist = annotations["annolist"][i_img]
    annorects = annolist["annorect"]
    if isinstance(annorects, dict): # Only one annorect (one person)
        if "annopoints" in annorects.keys():
            point = annorects["annopoints"]["point"]
            # Check that every joint we care about is annotated
            all_present = True
            for j in ALLOWED_JOINTS:
                all_present &= j in [p["id"] for p in point]
            if all_present:
                joints = []
                for j in ALLOWED_JOINTS:
                    joints.append([[p["x"], p["y"]] for p in point if p["id"] == j][0])
                annorect_imgs.append({"id": i_img, "joints": joints})

print(f"Indexed {len(annorect_imgs)} images with valid annotations")

# Copy those images to the valid dir
import shutil
annotation_output = []

for idx, i in enumerate(annorect_imgs):
    img_name = annotations["annolist"][i["id"]]["image"]["name"]
    joints = i["joints"]
    annotation_output.append(joints)
    shutil.copy(data_dir + "/images/" + img_name, data_dir + "/images/single/" + f"{idx}.jpg")

dump(annotation_output, Path(data_dir + "/images/single/annotations.json").open("w"))