import json

captions_path = "../../data/coco/annotations/captions_val2017.json"
# instances_path = "../../data/coco/annotations/instances_val2017.json"

captions = json.load(open(captions_path))
# instances = json.load(open(instances_path))

filename_id_dict = {image["file_name"]: image["id"] for image in captions["images"]}
id_caption_dict = {image["image_id"]: image["caption"] for image in captions["annotations"]}

def get_caption(image_filename):
    image_id = filename_id_dict[image_filename]
    return id_caption_dict[image_id]