from typing import List, Any
from math import log2

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
        if precision + recall  == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def mean_average_precision(precisions: List[float]) -> float:
        return sum(precisions) / len(precisions)

    @staticmethod
    def mrr(predictions_list: List[Any], ground_truth_list: List[Any]) -> float:
        # Mean Reciprocal Rank
        mrr_total = 0.0
        num_queries = len(predictions_list)
        
        for predictions, ground_truth in zip(predictions_list, ground_truth_list):
            for i, prediction in enumerate(predictions):
                if prediction in ground_truth:
                    mrr_total += 1 / (i + 1)
                    break
        return mrr_total / num_queries if num_queries > 0 else 0.0

    @staticmethod
    def calculate_relevance_scores(predictions, ground_truth):
        """
        Calculate the relevance scores of the retrieved files.

        Args:
            retrieved_files (list): The retrieved files.
            actual_files (list): The actual files that should be retrieved.

        Returns:
            list: The relevance scores of the retrieved files.
        """
        return [1 if file in ground_truth else 0 for file in predictions]

    @staticmethod
    def dcg(relevance_scores, k):
        """
        Calculate the Discounted Cumulative Gain (DCG) at rank k.

        Args:
            relevance_scores (list): The relevance scores of the results.
            k (int): The rank.

        Returns:
            float: The DCG at rank k.
        """
        return sum(
            rel / log2(i + 2) for i, rel in enumerate(relevance_scores[:k])
        )
       
    @staticmethod 
    def idcg(relevance_scores, k):
        """
        Calculate the Ideal Discounted Cumulative Gain (IDCG) at rank k.

        Args:
            relevance_scores (list): The relevance scores of the results.
            k (int): The rank.

        Returns:
            float: The IDCG at rank k.
        """
        return Metrics.dcg(sorted(relevance_scores, reverse=True), k)
    
    @staticmethod
    def ndcg(predictions, ground_truth, k):
        """
        Calculate the Normalized Discounted Cumulative Gain (NDCG) at rank k.

        Args:
            relevance_scores (list): The relevance scores of the results.
            k (int): The rank.

        Returns:
            float: The NDCG at rank k.
        """
        relevance_scores = Metrics.calculate_relevance_scores(predictions, ground_truth)
        dcg_score = Metrics.dcg(relevance_scores, k)
        idcg_score = Metrics.idcg(relevance_scores, k)
        if idcg_score == 0:
            return 0.0
        return dcg_score / idcg_score