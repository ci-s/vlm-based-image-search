import faiss
from sentence_transformers import SentenceTransformer

class FaissService:
    def __init__(self, encoder_model):
        self.encoder_model = SentenceTransformer(encoder_model)
        self.index = None

    def _encode_sentences(self, sentences):
        sentence_embeddings = self.encoder_model.encode(sentences, normalize_embeddings=True)
        return sentence_embeddings

    def create_index(self, sentences):
        if sentences is None:
            raise ValueError("Sentences cannot be None")
        
        sentence_embeddings = self._encode_sentences(sentences)
        d = sentence_embeddings.shape[1]
        index = faiss.IndexFlatIP(d) #TODO: parametrize
        index.add(sentence_embeddings)
        print("Index created with {} sentences".format(index.ntotal))
        self.index = index
        return self.index

    def search_index(self, query, k, threshold=None, increment_factor=2):
        
        query_embedding = self._encode_sentences([query])
        
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
        