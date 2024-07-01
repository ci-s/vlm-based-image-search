import os
from PIL import Image
from torch.utils.data import Dataset
from natsort import natsorted
from .get_coco import get_image_pretransformer, load_cocos_like_dataset

class CustomImageDataset(Dataset):
    def __init__(self, directory_path, model_name, isSorted = False, transform=None):
        self.directory_path = directory_path
        if transform is None:
            self.transform = get_image_pretransformer(model_name)
        else:
            self.transform = transform
        self.filenames = [f for f in os.listdir(directory_path) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        if isSorted:
            self.filenames = natsorted(self.filenames)
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.directory_path, self.filenames[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image
    
    def get_filename(self, idx):
        return self.filenames[idx]
    
class CocosCaptionDataset(Dataset):
    def __init__(self, captions_per_img, model_name, img_root = None, ann_root=None, option = "all"):
        self.captions = get_captions_from_dataset(model_name, captions_per_img, img_root, ann_root, option)
    def __len__(self):
        return len(self.captions)
    def __getitem__(self, idx):
        return self.captions[idx]
    def get_captions_in_range(self, start_index = 0, end_index = 500):
        return self.captions[start_index:end_index]
    
    
def get_captions_from_dataset(model_name, captions_per_img = 5, img_root = None, ann_root=None, option = "all"):
    dataset = load_cocos_like_dataset(captions_per_img, model_name, img_root, ann_root)
    caption_list = []
    index_lambda = None
    if option == "all":
        index_lambda = lambda capts : capts
    elif option == "first":
        index_lambda = lambda capts : [capts[0]]
    elif option == "concat":
        index_lambda = lambda capts: [" ".join(capts)]
    
    for _, captions in dataset:
        caption_list.extend(index_lambda(captions))
    return caption_list