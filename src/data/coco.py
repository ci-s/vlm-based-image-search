import json
import os
import sys

sys.path.append("../")
from services.settings import settings

captions_path = os.path.join(settings.data_dir, "coco/annotations/captions_val2017.json")
# instances_path = "../../data/coco/annotations/instances_val2017.json"
coco_url_prefix = "http://images.cocodataset.org/val2017/"

captions = json.load(open(captions_path))
# instances = json.load(open(instances_path))

filename_id_dict = {image["file_name"]: image["id"] for image in captions["images"]}
id_caption_dict = {image["image_id"]: image["caption"] for image in captions["annotations"]}

def get_caption(image_filename):
    image_id = filename_id_dict[image_filename]
    return id_caption_dict[image_id]

def get_url(image_filename):
    return coco_url_prefix + image_filename