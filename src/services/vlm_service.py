from .vlm_models_service import CLIPModel, LlavaModel, GITModel
from .database_service import DatabaseService
from torch.utils.data import DataLoader, Subset
import faiss
import torch


class EmbeddingService:
    def __init__(self, model_name, dataset, dataset_name, 
                 index_suffix=None, text_suffix=None, caption_suffix=None, model_parameters=None):
        self.model_name = model_name
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.database_service = DatabaseService(dataset_name, index_suffix, text_suffix, caption_suffix)
        
        if self.model_name == "CLIP":
            self.model = CLIPModel(self.model_name, model_parameters)
        elif self.model_name == "Llava":
            self.model = LlavaModel(self.model_name, model_parameters)
        elif self.model_name == "GIT":
            self.model = GITModel(self.model_name, model_parameters)
    
    @torch.no_grad()
    def get_embeddings(self, batch_size=16):
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        img_embeddings, text_embeddings = self._run_models(dataloader)
        index = self.database_service.create_index(img_embeddings)
        self._save_outputs(index, text_embeddings)
        if self.model_name != "CLIP":
            self.database_service.save_captions(self.model.get_captions())
        return index, text_embeddings
    
    def get_captions(self):
        return self.model.get_captions()    
    
    @torch.no_grad()
    def _run_models(self, dataloader):
        image_embeddings, text_embeddings = self.model.encode_images_texts(dataloader)
        return image_embeddings, text_embeddings

    def get_embeddings_with_checkpoint(self, batch_size=16, checkpoint_size=64, check_database=False):
        if checkpoint_size < batch_size:
            print("The size of the checkpoint should be more than batch size!")
            return
        start_index = 0
        dataset_len = len(self.dataset)
        index = None
        caption_embedding = []
        all_text_embedding = []
        if check_database and self.database_service.is_img_embd_exists(self.database_service._get_img_index_path):
            index = self.database_service.load_imgs_index()
            start_index = max(index.ntotal, 0)
        if check_database and self.database_service.is_text_embd_exists(self.database_service._get_text_embd_path):
            all_text_embedding = [self.database_service.load_texts_embeddings()]
        while start_index < dataset_len - 1:
            stop_index = min(dataset_len, start_index + checkpoint_size)
            dataloader = DataLoader(Subset(self.dataset, list(range(start_index, stop_index))), batch_size=batch_size, shuffle=False)
            image_embeddings, text_embeddings = self._run_models(dataloader)
            if index is None:
                index = faiss.IndexFlatIP(image_embeddings.shape[1])
            index.add(image_embeddings.cpu())
            all_text_embedding.append(text_embeddings)
            if self.model_name != "CLIP":
                caption_embedding.extend(self.model.get_captions())
                self.database_service.save_captions(caption_embedding)
            self._save_outputs(index, torch.cat(all_text_embedding, dim=0))
            start_index = stop_index
        return index, all_text_embedding
    
    def _save_outputs(self, index, text_embeddings):
        self.database_service.save_index(index)
        self.database_service.save_texts_embeddings(text_embeddings)






# class EmbeddingService:
#     def __init__(self, model_name, dataset, dataset_name, 
#                  index_suffix = None, text_suffix = None, caption_suffix = None, model_parameters = None):
#         self.model_name = model_name
#         self.dataset = dataset
#         self.dataset_name = dataset_name
#         self.database_service = DatabaseService(dataset_name, index_suffix, text_suffix, caption_suffix)
        
#         if self.model_name == "CLIP":
#             self.model = CLIPModel(self.model_name, model_parameters)
#         elif self.model_name == "Llava":
#             self.model = LlavaModel(self.model_name, model_parameters)
#         elif self.model_name == "GIT":
#             self.model = GITModel(self.model_name, model_parameters)
    
#     @torch.no_grad()            
#     def get_embeddings(self, batch_size = 16):
#         dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
#         img_embeddings, text_embeddings = self._run_models(dataloader)
#         index = self.database_service.create_index(img_embeddings)
#         self._save_outputs(index, text_embeddings)
#         if (self.model_name is not "CLIP"):
#                 self.database_service.save_captions(self.model.get_captions())
#         return index, text_embeddings
    
#     def get_captions(self):
#         return self.model.get_captions()    
    
#     @torch.no_grad()    
#     def _run_models(self, dataloader):
#         image_embeddings, text_embeddings = self.model.encode_images_texts(dataloader)
#         return image_embeddings, text_embeddings

#     def get_embeddings_with_checkpoint(self, batch_size = 16, checkpoint_size = 64, check_database = False):
#         if checkpoint_size < batch_size:
#             print("The size of the checkpoint should be more than batch size!")
#             return
#         start_index = 0
#         dataset_len = len(self.dataset)
#         index = None
#         caption_embedding = []
#         all_text_embedding = []
#         if check_database and self.database_service.is_img_embd_exists(self.database_service._get_img_index_path):
#             index = self.database_service.load_imgs_index()
#             start_index = max(index.ntotal, 0)
#         if check_database and self.database_service.is_text_embd_exists(self.database_service._get_text_embd_path):
#             all_text_embedding.append(self.database_service.load_texts_embeddings())
#         while(start_index < dataset_len - 1):
#             stop_index = min(dataset_len, start_index + checkpoint_size)
#             dataloader = DataLoader(Subset(self.dataset, [i for i in range(start_index, stop_index)]), batch_size = batch_size, shuffle = False)
#             image_embeddings, text_embeddings = self._run_models(dataloader)
#             if index is None:
#                 index = faiss.IndexFlatIP(image_embeddings.shape[1])
#             index.add(image_embeddings)
#             all_text_embedding.append(text_embeddings)
#             all_text_embedding = torch.cat(all_text_embedding, dim = 0)
#             if (self.model_name != "CLIP"):
#                 caption_embedding.extend(self.model.get_captions())
#                 self.database_service.save_captions(caption_embedding)
#             self._save_outputs(index, all_text_embedding)
#             start_index = stop_index
    
#     def _save_outputs(self, index, text_embeddings):
#         self.database_service.save_index(index)
#         self.database_service.save_texts_embeddings(text_embeddings)
    
            
            
            
            
            
            