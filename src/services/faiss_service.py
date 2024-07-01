import faiss
from typing import Any, List

class FaissService:
    def __init__(self, encoder_model): # encoder_model is either an instance of SentenceTransformer or CLIP or None = because directly query embeddings will be provided to search with
        if encoder_model is not None:
            self.encoder_model = encoder_model
        else:
            self.encoder_model = None
        self.index = None

    def _encode_sentences(self, sentences):
        sentence_embeddings = self.encoder_model.encode(sentences, normalize_embeddings=True)
        return sentence_embeddings

    def create_index(self, image_representation: List[Any]):
        if image_representation is None:
            raise ValueError("Image representations cannot be None")
        if isinstance(image_representation, List) and isinstance(image_representation[0], str):
            image_embeddings = self._encode_sentences(image_representation)
        else:
            image_embeddings = image_representation
        d = image_embeddings.shape[1]
        index = faiss.IndexFlatIP(d) #TODO: parametrize
        index.add(image_embeddings)
        print("Index created with {} sentences".format(index.ntotal))
        self.index = index
        return self.index

    def search_index(self, query, k, threshold=None, increment_factor=2):
        
        if isinstance(query, str):
            query_embedding = self._encode_sentences([query])
        else:
            query_embedding = query
        
        if not threshold: # return k results
            print("Returning top k results...")
            D, I = self.index.search(query_embedding, k)
            return D[0], I[0]
        else:
            print("Returning results with scores > threshold...")
            while True: # return results with scores > threshold
                D, I = self.index.search(query_embedding, k)
                scores = D[0]
                if len(scores) == 0 or scores[-1] <= threshold:
                    break
                k *= increment_factor
            mask = scores > threshold
            filtered_scores = scores[mask]
            filtered_indices = I[0][mask]
            return filtered_scores, filtered_indices
        