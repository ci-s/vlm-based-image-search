import argparse
import json
import random
random.seed(2024)
import sys

sys.path.append("../")
from data.coco import create_gt_captions_dict
from services.search import ImageCaptions, SearchService
from eval.metrics import Metrics

# Parse argumentsnts
parser = argparse.ArgumentParser()
parser.add_argument("--caption_path", type=str, help="Name of the predicted caption file, which is a JSON")
parser.add_argument("--mode", type=str, choices=['concat', 'random', 'first'], default='concat', help="Method to handle multiple captions in COCO")

args = parser.parse_args()
caption_path = args.caption_path
mode = args.mode


predicted_image_captions = ImageCaptions(json.load(open(caption_path)))

gt_dict = create_gt_captions_dict(predicted_image_captions.get_filenames(), mode)
gt_image_captions = ImageCaptions(gt_dict)

encoder_model_name = 'sentence-transformers/all-MiniLM-L6-v2'

# Search
ss = SearchService(encoder_model_name, predicted_image_captions) # Important: This should be modified for CLIP

# Get ground truth
gt_ss = SearchService(encoder_model_name, gt_image_captions)

print("Evaluating the model...")
query_list = ["birthday cake", # an object
            "children throwing snowballs", # an action
            "two dogs", # objects with a count
            "people getting ready to work", # reasoning
            ]

performance_dict = {"recall": [], "precision": [], "f1": [], "nDCG": [], "mRR": []}

for query in query_list:
    retrieved_files = ss.search(query)
    gt_retrieved_files = gt_ss.search(query)
    
    performance_dict["recall"].append(Metrics.recall(retrieved_files, gt_retrieved_files))
    performance_dict["precision"].append(Metrics.precision(retrieved_files, gt_retrieved_files))
    performance_dict["f1"].append(Metrics.f1_score(retrieved_files, gt_retrieved_files))
    performance_dict["nDCG"].append(Metrics.ndcg(retrieved_files, gt_retrieved_files, 5)) #TODO:parametrize
    performance_dict["mRR"].append(Metrics.mrr(retrieved_files, gt_retrieved_files))


for metric, values in performance_dict.items():
    print(f"{metric}: {sum(values)/len(values):.2f}")