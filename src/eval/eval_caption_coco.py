import argparse
import json
import os
import sys
sys.path.append("../")

from eval.text_similarity import calculate_similarity
from data.coco import get_caption, get_url
from services.settings import settings

# Assumptions
# The output file: predicted captions
# - The output file is a JSON file with the following structure:
# {
#     "filename1": "predicted_caption1",
#     "filename2": "predicted_caption2",
#     ...
# }
# (If you'd like it work with a different structure, you can modify the code and ADD it below.)
# The output file is located in the "outputs" directory.
# Evaluation result can be saved, containing the COCO url to display photos and examine performance.

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--output_file", type=str, help="Name of the output file")
parser.add_argument("--num_samples", type=int, help="Number of the samples that will be evaluated", default=500)
parser.add_argument("--save_result", type=bool, help="Whether to save the eval result", default=False)

args = parser.parse_args()
output_file = args.output_file
save_result = args.save_result
# Load the output file
response = json.load(open(os.path.join(settings.output_dir, output_file)))

# Calculate similarity
sim_scores = {}
for filename, predicted_caption in response.items():
    caption = get_caption(filename)
    similarity = calculate_similarity(caption, predicted_caption)
    sim_scores[get_url(filename)] = similarity
    
# Calculate the average similarity
average_similarity = sum(sim_scores.values()) / len(sim_scores)
print(f"Average similarity: {average_similarity:.2f}")

if save_result:
    result = {
        "average_similarity": average_similarity,
        "similarity_scores": sim_scores
    }
    with open(os.path.join(settings.output_dir, "eval_result_"+output_file.split(".")[0]+".json"), "w") as f:
        json.dump(result, f)