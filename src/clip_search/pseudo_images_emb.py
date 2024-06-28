import sys
sys.path.append("..")
import os
from data.custom_dataset import CustomImageDataset
from services import vlm_models_service as models
from torch.utils.data import DataLoader
from services.settings import settings
from services.data_utilizer import DataUtilizer
data_dir = settings.data_dir
dataset_n = "nocaps"
dataset_path = dataset_n + "/generated_images"
images_directory = os.path.join(data_dir, dataset_path)
output_dir = settings.output_dir

def generate_embeddings_images(output_prefix, dataset_name, directory_path = images_directory, save_in_Index = False, model_name = "CLIP", batch_size = 64, isSorted=False):
    dataset = CustomImageDataset(directory_path, model_name, isSorted)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    data_utilizer = DataUtilizer(dataset_name)
    image_embeddings = None
    captions = None
    is_Caption = False
    if model_name == "CLIP":
        model = models.CLIPModel("CLIP")
    elif model_name == "Llava":
        model = models.LlavaModel("Llava")
        is_Caption = True
    elif model_name == "GIT":
        model = models.LlavaModel("GIT")
        is_Caption = True
    else:
        raise ValueError("model names should be CLIP, Llava or GIT")
    image_embeddings = model.encode_images(dataloader)
    output_path = os.path.join(output_dir, output_prefix)
    prefix = output_prefix + dataset_name
    if save_in_Index: 
        index = data_utilizer.create_index(image_embeddings)
        output_path_ind = os.path.join(output_dir, prefix + ".index")
        data_utilizer.save_index(index, output_path_ind)
    else:
        output_path = os.path.join(output_dir, prefix + "_embeddings.pth")
        data_utilizer.save_texts_embeddings(image_embeddings, output_path)
    if is_Caption:
        captions = model.get_captions()
        output_path = os.path.join(output_dir, prefix + "_captions.pth")
        data_utilizer.save_texts_embeddings(captions, output_path)

generate_embeddings_images("generated_images", "mscoco", isSorted=True)