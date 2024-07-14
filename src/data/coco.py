import json
import os
import random
import sys

sys.path.append("../")
from services.settings import settings

captions_path = os.path.join(settings.data_dir, "coco-images/annotations/captions_val2017.json")
# instances_path = "../../data/coco/annotations/instances_val2017.json"
coco_url_prefix = "http://images.cocodataset.org/val2017/"

captions = json.load(open(captions_path))
# instances = json.load(open(instances_path))

filename_id_dict = {image["file_name"]: image["id"] for image in captions["images"]}

id_caption_dict = {}
for image in captions["annotations"]:
    image_id = image["image_id"]
    caption = image["caption"]
    if image_id in id_caption_dict:
        id_caption_dict[image_id].append(caption)
    else:
        id_caption_dict[image_id] = [caption]

def get_captions(image_filename):
    image_id = filename_id_dict[image_filename]
    return id_caption_dict[image_id]

def get_url(image_filename):
    return coco_url_prefix + image_filename

# gt = ground truth
def create_gt_captions_dict(files, mode): 
    # To handle multiple captions in COCO, select a method
    match mode:
        case 'concat':
            return {file: ' '.join(get_captions(file)) for file in files}
        case 'random':
            return {file: random.choice(get_captions(file)) for file in files}
        case 'first':
            return {file: get_captions(file)[0] for file in files}
        case 'avg_embedding':
            # join all captions but add a / between them
            return {file: '/'.join(get_captions(file)) for file in files}