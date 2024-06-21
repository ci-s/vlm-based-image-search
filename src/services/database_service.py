import faiss
import torch
import json
import sys
import os


sys.path.append("../../outputs")
img_index_suffix_def = "_images.index"
text_embd_suffix_def= "_texts_emb.pth"
caption_suffix_def = "_captions.pth"
path_prefix_def = "../../outputs/"


class DatabaseService:
    def __init__(self, dataset_n, path_prefix = None, img_index_suffix = None, text_embd_suffix = None, caption_suffix = None):
        self.db = dataset_n
        if path_prefix is None:
            path_prefix = path_prefix_def
        if img_index_suffix is None:
            img_index_suffix= img_index_suffix_def
        if text_embd_suffix is None:
            text_embd_suffix = text_embd_suffix_def
        if caption_suffix is None:
            caption_suffix = caption_suffix_def
            
        self.img_index_path = path_prefix + "_" + self.db + img_index_suffix
        self.text_embd_path = path_prefix + "_" + self.db + text_embd_suffix
        self.caption_path = path_prefix + "_" + self.db + caption_suffix
    
    def _get_img_index_path(self):
        return self.img_index_path
    
    def _get_text_embd_path(self):
        return self.text_embd_path
    
    def _get_caption_path(self):
        return self.caption_path
    
    def create_index(self, embedding):
        index = faiss.IndexFlatIP(embedding.shape[1])
        index.add(embedding.cpu())
        print(f"Index is created with {index.ntotal} embeddings")
        return index

    def save_index(self, index, index_path=None):
        if index_path is None:
            index_path = self._get_img_index_path()
        faiss.write_index(index, index_path)
    
    def load_imgs_index(self, index_path=None):
        if index_path is None:
            index_path = self._get_img_index_path()
        index = faiss.read_index(index_path)
        return index

    def save_texts_embeddings(self, embedding, text_embd_path=None):
        if text_embd_path is None:
            text_embd_path = self._get_text_embd_path()
        torch.save(embedding, text_embd_path)

    def load_texts_embeddings(self, text_embd_path=None, device = "cpu"):
        if text_embd_path is None:
            text_embd_path = self._get_text_embd_path()
        return torch.load(text_embd_path, map_location=torch.device(device))

    def save_results(self, results, result_path=None):
        if result_path is None:
            result_path = self.db + "metrics.json"
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=4)

    def is_img_embd_exists(self, index_path=None):
        if index_path is None:
            index_path = self._get_img_index_path()
        return os.path.exists(index_path)

    def is_text_embd_exists(self, text_embd_path=None):
        if text_embd_path is None:
            text_embd_path = self._get_text_embd_path()
        return os.path.exists(text_embd_path)
    
    def save_captions(self, captions, captions_path=None):
        if captions_path is None:
            captions_path = self._get_caption_path()
        torch.save(captions, captions_path)

    def load_caption(self, captions_path=None, device = "cpu"):
        if captions_path is None:
            captions_path = self._get_caption_path()
        return torch.load(captions_path, map_location=torch.device(device))
    
    



