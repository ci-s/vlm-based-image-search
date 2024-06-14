import faiss

class FaissService:
    def __init__(self, encoder_model):
        self.encoder_model = encoder_model
        self.index = None

    def _encode_sentences(self, sentences):
        sentence_embeddings = self.encoder_model.encode(sentences)
        return sentence_embeddings

    def create_index(self, sentences):
        sentence_embeddings = self._encode_sentences(sentences)
        d = sentence_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(sentence_embeddings)
        print("Index created with {} sentences".format(index.ntotal))
        self.index = index
        return self.index

    def search_index(self, query, k):
        query_embedding = self._encode_sentences([query])
        D, I = self.index.search(query_embedding, k)
        return D, I