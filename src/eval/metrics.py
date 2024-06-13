from typing import List, Any

class Metrics:
    @staticmethod
    def recall(predictions: List[Any], ground_truth: List[Any]) -> float:
        true_positives = len(set(predictions) & set(ground_truth))
        false_negatives = len(set(ground_truth) - set(predictions))
        return true_positives / (true_positives + false_negatives)

    @staticmethod
    def precision(predictions: List[Any], ground_truth: List[Any]) -> float:
        true_positives = len(set(predictions) & set(ground_truth))
        false_positives = len(set(predictions) - set(ground_truth))
        return true_positives / (true_positives + false_positives)

    @staticmethod
    def f1_score(predictions: List[Any], ground_truth: List[Any]) -> float:
        precision = Metrics.precision(predictions, ground_truth)
        recall = Metrics.recall(predictions, ground_truth)
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def mean_average_precision(precisions: List[float]) -> float:
        return sum(precisions) / len(precisions)
    
    @staticmethod
    def normalized_discounted_cumulative_gain(relevance_scores: List[float]) -> float:
        return sum(relevance_scores) / len(relevance_scores)
    
    @staticmethod
    def mean_reciprocal_rank(rankings: List[int]) -> float:
        return sum([1/ranking for ranking in rankings]) / len(rankings)
