import sys
sys.path.append("..")
from services.vlm_models_service import SearchModel 
from data.get_coco import load_cocos_like_dataset_in_range
from torch.utils.data import DataLoader
from services.data_utilizer import DataUtilizer


def generate_embeddings(model_name, dataset_n, img_root = None, ann_root = None):
    model = SearchModel(model_name).get_model()
    dataset = load_cocos_like_dataset_in_range(5, model_name, img_root, ann_root, "first")
    dataloader = DataLoader(dataset, batch_size = 64, shuffle= False)
    image_embeddings, text_embeddings = model.encode_images_texts(dataloader)
    data_utilizer = DataUtilizer(dataset_n)
    output_dir = "../../outputs/"
    suffix = dataset_n + "_" + model_name + "_500_" + "first.pth"
    image_path = output_dir + "image_" + suffix
    text_path = output_dir + "texts_" + suffix
    data_utilizer.save_texts_embeddings(image_embeddings, image_path)
    data_utilizer.save_texts_embeddings(text_embeddings, text_path)
        
models = ["CLIP", "GIT", "Llava"]
img_root = "../../data/nocaps/validation_images"
ann_root = "../../data/nocaps/no_caps.json"
for model in models:
    generate_embeddings(model, "nocaps", img_root, ann_root)

