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
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def mean_average_precision(precisions: List[float]) -> float:
        return sum(precisions) / len(precisions)
    
    @staticmethod
    def mrr(predictions: List[Any], ground_truth: List[Any]) -> float:
        # Mean Reciprocal Rank
        rankings = []
        for i, prediction in enumerate(predictions):
            if prediction in ground_truth:
                rankings.append(1 / (i + 1))
        return sum([1/ranking for ranking in rankings]) / len(rankings)

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
        return Metrics.dcg(relevance_scores, k) / Metrics.idcg(relevance_scores, k)