import sys 
sys.path.append("..")
sys.path.append(".")
from clip_search.evaluate_models import evaluate_models
from services.settings import settings
import os
from services.data_utilizer import DataUtilizer
from data.get_coco import load_cocos_like_dataset_in_range

out_dir = settings.output_dir
prefix_image= "image_"
prefix_text = "texts_"
prefix_gen = "generated_images_"
#suffix= "_500_last.pth"
suffix = "_500_first.pth"
data_util = DataUtilizer("")
dataset = load_cocos_like_dataset_in_range(5, "CLIP", option = "first")
filenames = [dataset.dataset.get_filename(i) for i in range(500)]

models_map = {"CLIP" : {},
              "Llava": {},
              "GIT": {}
              }
for dataset_name in ["mscoco_", "nocaps_"]:
    for model in ["CLIP", "Llava", "GIT"]:
        models_map[model]["image_path_" + dataset_name] = os.path.join(out_dir, prefix_image + dataset_name + model + suffix)
        models_map[model]["text_path_" + dataset_name] = os.path.join(out_dir, prefix_text + dataset_name + model + suffix)
        models_map[model]["gen_image_" + dataset_name] = os.path.join(out_dir, prefix_gen + dataset_name + model + "_embeddings.pth")

results = {}
for dataset_name in ["mscoco_", "nocaps_"]:
    for model, maps in models_map.items():
        image_embd = data_util.load_texts_embeddings(maps["image_path_" + dataset_name]).cpu()
        text_embd = data_util.load_texts_embeddings(maps["text_path_" + dataset_name]).cpu()
        gen_embd = data_util.load_texts_embeddings(maps["gen_image_" + dataset_name]).cpu()
        for alpha in [0.0,  0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 1.0]:
            result_emb = gen_embd * alpha + (1-alpha) * text_embd
            for k in [3, 5, 10]:
                result = evaluate_models(image_embd, result_emb, filenames, k)
                results[f"{model}-{dataset_name}-{alpha}-{k}"] = result
data_util.save_results(results, "../../outputs/" + "multimodal_all_results.json")
