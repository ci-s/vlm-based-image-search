import os
import json
import sys

sys.path.append("../")
from services.settings import settings

def first_n_files(n: int) -> list[str]:
    file_restriction = os.path.join(settings.data_dir, "coco/annotations/captions_val2017.json")
    file_restriction = json.load(open(file_restriction))

    file_list = []
    for d in file_restriction["images"][:n]:
        file_list.append(d["file_name"])
        
    return file_list

def merge_json_files(file_paths, output_path):
    merged_data = {}

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)
            merged_data.update(data)

    with open(output_path, 'w') as output_file:
        json.dump(merged_data, output_file, indent=4)
    
    return merged_data