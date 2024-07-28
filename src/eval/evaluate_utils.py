import torch


def dcg(relevance_scores):
 
    return sum((rel / torch.log2(torch.tensor(idx + 2).float())) for idx, rel in enumerate(relevance_scores))

def ndcg(correct):
    ndcgs = []
    for i in range(correct.size(0)):
        relevance_scores = correct[i]
        sorted_relevance_scores = torch.sort(relevance_scores, descending=True).values
        
        # Compute DCG for the given relevance scores
        actual_dcg = dcg(relevance_scores)
        
        # Compute IDCG for the ideal case
        ideal_dcg = dcg(sorted_relevance_scores)
        
        # Compute nDCG
        if ideal_dcg == 0:
            ndcg_value = 0.0
        else:
            ndcg_value = actual_dcg / ideal_dcg
        
        ndcgs.append(ndcg_value)
    
    return sum(ndcgs) / len(ndcgs)

def precision(correct):
    precisions = []
    for i in range(correct.size(0)):
        relevant_count = correct[i].sum().item() 
        retrieved_count = correct.size(1) 
        if retrieved_count > 0:
            precision_at_k = relevant_count / retrieved_count
            precisions.append(precision_at_k)
        else:
            precisions.append(0.0)
    return sum(precisions) / len(precisions)

def mean_reciprocal_rank(correct):
    correct = correct.to(dtype=torch.bool)
    reciprocal_ranks = torch.zeros(correct.size(0), dtype=torch.float32)
    for i in range(correct.size(0)):
        if correct[i].any():
            first_relevant_index = correct[i].nonzero(as_tuple=True)[0][0].item()
            reciprocal_ranks[i] = 1.0 / (first_relevant_index + 1)
    return reciprocal_ranks.mean().item()

def average_precision(correct):
    precisions = []
    for i in range(correct.size(0)):
        relevant_indices = correct[i].nonzero(as_tuple=True)[0]
        if relevant_indices.numel() > 0:
            ap = 0.0
            for idx in range(relevant_indices.size(0)):
                precision_at_k = (relevant_indices[:idx+1].numel()) / (relevant_indices[idx].item() + 1)
                ap += precision_at_k
            precisions.append(ap / relevant_indices.size(0))
        else:
            precisions.append(0.0)
    return sum(precisions) / len(precisions)

def evaluate(img_index, text_embeddings, text_to_image_map, k_vals = [1, 3, 5, 10, 20, 50]):
    mrr_values = []
    average_precision_values = []
    recall_values = []
    ndcg_values = []
    results = {'MRR@k': {}, 'Average Precision': {}, 'Recall' : {}, 'NDCG' : {}}
    for k in k_vals:
        # Extract top k indices
        # text_embeddings.to(device)
        # img_index.to(device)
        _, I = img_index.search(text_embeddings.to("cpu"), k)
        I = torch.from_numpy(I).to("cpu")
        text_to_image_map = text_to_image_map.to("cpu")
        correct = torch.eq(I, text_to_image_map.unsqueeze(1))
        
        # Mean Reciprocal Rank (MRR)
        mrr = mean_reciprocal_rank(correct)
        mrr_values.append(mrr)
        
        # Mean Average Precision (MAP)
        map_value = average_precision(correct)
        average_precision_values.append(map_value)
        ndcg_value_at_k = ndcg(correct)
        ndcg_values.append(ndcg_value_at_k)
        # Recall@k
        recall_at_k = correct.any(dim=1).float().mean().item()
        recall_values.append(recall_at_k)
        results["MRR@k"][f"MRR@{k}"] = mrr
        results["Average Precision"][f"av_prec@{k}"] = map_value
        results["Recall"][f"recall@{k}"] = recall_at_k
        results["NDCG"][f"ndcg@{k}"] = ndcg_value_at_k
    return results

def evaluate_objects(img_index, text_embeddings, text_to_image_map, k_vals = [1, 3, 5, 10, 20, 50]):
    mrr_values = []
    average_precision_values = []
    recall_values = []
    ndcg_values = []
    results = {'MRR@k': {}, 'Average Precision': {}, 'Recall' : {}, 'NDCG' : {}, "f1": {}}
    for k in k_vals:
        _, I = img_index.search(text_embeddings.to("cpu"), k)
        I = torch.from_numpy(I).to("cpu")
        text_to_image_map = text_to_image_map.to("cpu")
        correct = find_correctness(I, text_to_image_map)
        
        # Mean Reciprocal Rank (MRR)
        mrr = mean_reciprocal_rank(correct)
        mrr_values.append(mrr)
        
        # Mean Average Precision (MAP)
        map_value = average_precision(correct)
        average_precision_values.append(map_value)
        ndcg_value_at_k = ndcg(correct).item()
        ndcg_values.append(ndcg_value_at_k)
        # Recall@k
        recall_at_k = correct.any(dim=1).float().mean().item()
        recall_values.append(recall_at_k)
        results["MRR@k"][f"MRR@{k}"] = round_values(mrr)
        f1 =2*recall_at_k*map_value / (recall_at_k+map_value)
        results["f1"][f"f1@{k}"] = round_values(f1)
        results["Average Precision"][f"av_prec@{k}"] = round_values(map_value)
        results["Recall"][f"recall@{k}"] = round_values(recall_at_k)
        results["NDCG"][f"ndcg@{k}"] = round_values(ndcg_value_at_k)
    return results

def round_values(value):
    return round(value, 3)

def find_correctness(index_list, text_map):
    correct = list()
    for index_l, text_m in zip(index_list, text_map):
        correct.append([text_m[ind] == 1 for ind in index_l])
    return torch.tensor(correct)

def print_results(results, k_vals = [1, 3, 5, 10, 20, 50]):
    for k in k_vals:
        mrr = results["MRR@k"][f"MRR@{k}"]
        ap = results["Average Precision"][f"av_prec@{k}"]
        recall = results["Average Precision"][f"av_prec@{k}"]
        ndcg_val = results["NDCG"][f"ndcg@{k}"]

        print (f"MRR@{k}: {mrr}")
        print(f"av_prec@{k}: {ap}")
        print(f"call@{k}: {recall}")
        print(f"ndcg@{k}: {ndcg_val}")
        print("*"*20)