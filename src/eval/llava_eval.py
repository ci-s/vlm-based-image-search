import json
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import random
random.seed(2024)
import sys

sys.path.append("../")
from data.coco import create_gt_captions_dict
from services.search import ImageRepresentations, SearchService
from eval.metrics import Metrics
from services.settings import settings

caption_path = os.path.join(settings.output_dir,"response_dict.json") # TODO merge if needed

llava_predicted_file = json.load(open(caption_path))

# Get first 500
file_restriction = os.path.join(settings.data_dir, "coco/annotations/captions_val2017.json")
file_restriction = json.load(open(file_restriction))

file_list = []
for d in file_restriction["images"][:500]:
    file_list.append(d["file_name"])
    
llava_predicted_file = {filename: caption for filename, caption in llava_predicted_file.items() if filename in file_list}

print("Length of the file list: ", len(file_list))
print("Length of the predicted files: ", len(llava_predicted_file))


llava_image_representions = ImageRepresentations(filenames=list(llava_predicted_file.keys()), representations=list(llava_predicted_file.values()), url_prefix="http://images.cocodataset.org/val2017/")

encoder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

output_filename = "llava_eval_0" # TODO
queries_path = "/usr/prakt/s0070/vlm-based-image-search/src/eval/extended_queries_w_keywords.json" # TODO
query_dict = json.load(open(queries_path))["categories"]


experiment = {}
for mode, k, threshold in [("concat", 10, None), ("concat", 10, 0.3),
                           ("random", 10, None), ("random", 10, 0.3),
                           ("first", 10, None), ("first", 10, 0.3),
                           ]:
    print(f"Mode: {mode}, k: {k}, threshold: {threshold}")
    # Get results
    gt_dict = create_gt_captions_dict(llava_image_representions.get_filenames(), mode)
    gt_image_captions = ImageRepresentations(list(gt_dict.keys()), list(gt_dict.values()), url_prefix="http://images.cocodataset.org/val2017/")
    ss = SearchService(llava_image_representions, encoder_model, threshold=threshold) 
    gt_ss = SearchService(gt_image_captions, encoder_model, threshold=threshold)

    print("Evaluating the model...")

    overall_performance_summary = {}

    for category, queries in query_dict.items():
        category_performance_dict = {"recall": [], "precision": [], "f1": [], "nDCG": [], "mRR": []}
        
        for query in queries:
            retrieved_files = ss.search(query)
            gt_retrieved_files = gt_ss.search(query)
            # print(f"The number of files predicted: {len(retrieved_files)}")
            # print(f"The number of files in the ground truth: {len(gt_retrieved_files)}")
            
            category_performance_dict["recall"].append(Metrics.recall(retrieved_files, gt_retrieved_files))
            category_performance_dict["precision"].append(Metrics.precision(retrieved_files, gt_retrieved_files))
            category_performance_dict["f1"].append(Metrics.f1_score(retrieved_files, gt_retrieved_files))
            category_performance_dict["nDCG"].append(Metrics.ndcg(retrieved_files, gt_retrieved_files, 5)) #TODO:parametrize
            category_performance_dict["mRR"].append(Metrics.mrr(retrieved_files, gt_retrieved_files))
            
        overall_performance_summary[category] = {}
        for metric, values in category_performance_dict.items():
            overall_performance_summary[category][metric] = sum(values)/len(values)
            
    # Initialize a dictionary to hold the cumulative values
    cumulative_metrics = {"recall": 0, "precision": 0, "f1": 0, "nDCG": 0, "mRR": 0}
    num_categories = len(overall_performance_summary)

    # Accumulate the values
    for category, metrics in overall_performance_summary.items():
        for metric, value in metrics.items():
            cumulative_metrics[metric] += value

    # Calculate the mean for each metric
    overall_performance_summary["overall"] = {}
    for metric, cumulative_value in cumulative_metrics.items():
        overall_performance_summary["overall"][metric] = cumulative_value / num_categories
        
    experiment[f"{mode}_{k}_{threshold}"] = overall_performance_summary
    
json.dump(experiment, open(os.path.join(settings.output_dir,output_filename+".json"), "w"), indent=4)

# Get dataframe
# Prepare data for the DataFrame
data = []
for key, performance_summary in experiment.items():
    mode, k, threshold = key.split('_')
    search_method = 'top 10' if threshold == 'None' else f'threshold > {threshold}'
    for category, metrics in performance_summary.items():
        row = {
            'COCO Aggregation method': mode,
            'Search method': search_method,
            'Query Category': category,
            'Avg Precision': metrics['precision'],
            'Avg Recall': metrics['recall'],
            'Avg F1': metrics['f1'],
            'nDCG': metrics['nDCG'],
            'mRR': metrics['mRR']
        }
        data.append(row)

# Create the DataFrame
df = pd.DataFrame(data)

# Reorder the columns to match the provided image
df = df[['COCO Aggregation method', 'Search method', 'Query Category', 'Avg Precision', 'Avg Recall', 'Avg F1', 'nDCG', 'mRR']]
df.sort_values(by=["COCO Aggregation method", "Query Category", "Search method"], inplace=True)
# Save the DataFrame to a CSV file
df.to_csv(os.path.join(settings.output_dir,output_filename+".csv"), index=False)