# run python app.py or gradio app.py (to reload the app on changes automatically)

import gradio as gr
from sentence_transformers import SentenceTransformer
import json
import os
import sys

sys.path.append("../")
sys.path.append(".")

from services.data_utilizer import DataUtilizer
from demo.clip import get_text_embedding
from data.coco import create_gt_captions_dict
from services.search import ImageRepresentations, SearchService
from services.settings import settings
from demo.utils import format_score

coco_path = os.path.join(settings.data_dir, "coco/images/val2017/")

k_default = 10
mode = "random"

# LLAVA
llava_caption_path = os.path.join(settings.output_dir, "response_dict.json") # format-> filename:caption
llava_predicted_file = json.load(open(llava_caption_path))

llava_image_representions = ImageRepresentations(filenames=list(llava_predicted_file.keys()), representations=list(llava_predicted_file.values()), url_prefix="http://images.cocodataset.org/val2017/")
encoder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder=settings.models_cache_dir)

llava_ss = SearchService(image_representations=llava_image_representions, encoder_model=encoder_model, k=k_default) #threshold=0.3

# GIT
git_caption_path = os.path.join(settings.output_dir, "response_dict_git_500.json") 
git_predicted_file = json.load(open(git_caption_path))
git_image_representions = ImageRepresentations(filenames=list(git_predicted_file.keys()), representations=list(git_predicted_file.values()), url_prefix="http://images.cocodataset.org/val2017/")
git_ss = SearchService(image_representations=git_image_representions, encoder_model=encoder_model, k=k_default)

# define another search service i.e for CLIP, change image representations
coco_root = "/storage/group/dataset_mirrors/old_common_datasets/coco2017/coco_val17/val2017"
clip_image_path = os.path.join(settings.output_dir, "image_mscoco_CLIP_500_first.pth")
data_util = DataUtilizer("")
clip_image_repr = data_util.load_texts_embeddings(clip_image_path).cpu()
clip_filename_path = os.path.join(settings.output_dir, "filenames_clip.json")
clip_filenames = data_util.load_json_file(clip_filename_path)
clip_image_representations = ImageRepresentations(clip_filenames, clip_image_repr, url_prefix="http://images.cocodataset.org/val2017/")

clip_ss = SearchService(image_representations=clip_image_representations, k=k_default)

# COCO captions as Ground truth
gt_dict = create_gt_captions_dict(llava_image_representions.get_filenames(), mode) # TODO: get filenames from somewhere else
gt_image_captions = ImageRepresentations(list(gt_dict.keys()), list(gt_dict.values()), url_prefix="http://images.cocodataset.org/val2017/")
gt_ss = SearchService(gt_image_captions, encoder_model, k=k_default)

# We can customize this function to use different models
def llava_search(query, k, threshold):
    retrieved_files, captions, scores = llava_ss.search_and_caption(query, k, threshold, return_url=False)
    return [(os.path.join(coco_root, file), format_score(score)+" "+caption) for file, caption, score in zip(retrieved_files, captions, scores)]

def git_search(query, k, threshold):
    retrieved_files, captions, scores = git_ss.search_and_caption(query, k, threshold, return_url=False)
    return [(os.path.join(coco_root, file), format_score(score)+" "+caption) for file, caption, score in zip(retrieved_files, captions, scores)]

def clip_search(query, k, threshold):
    retrieved_files, scores = clip_ss.search_and_score(get_text_embedding([query]), k, threshold, return_url=False)
    return [(os.path.join(coco_root, file), format_score(score)) for file, score in zip(retrieved_files, scores)] # No captions for this option

def gt_search(query, k, threshold):
    retrieved_files, captions, scores = gt_ss.search_and_caption(query, k, threshold, return_url=False)
    return [(os.path.join(coco_root, file), format_score(score)+" "+caption) for file, caption, score in zip(retrieved_files, captions, scores)]

def search(model_name, query, k, threshold):
    if model_name == "LLaVa":
        return llava_search(query, k, threshold)
    elif model_name == "CLIP":
        return clip_search(query, k, threshold)
    elif model_name == "GIT":
        return git_search(query, k, threshold)
    elif model_name == "Ground Truth":
        return gt_search(query, k, threshold)

k, threshold = None, None

with gr.Blocks() as demo:
    with gr.Column():
        with gr.Column(scale=1):
            model_choices = gr.CheckboxGroup(choices=["LLaVa", "CLIP", "GIT", "Ground Truth"], label="Select models")
            query = gr.Textbox(label="Enter your query here", scale = 1)
            mode = gr.Radio(
                choices=["k", "threshold"], 
                label="Choose search mode"
            )
            k_slider = gr.Slider(
                1, 50, step=1, label="Select k", visible=False
            )
            threshold_slider = gr.Slider(
                0.05, 0.95, step=0.01, label="Select threshold", visible=False
            )
            submit_button = gr.Button("Submit",scale = 1)
            
        def update_visibility(mode):
            if mode == "k":
                return gr.update(visible=True), gr.update(visible=False)
            elif mode == "threshold":
                return gr.update(visible=False), gr.update(visible=True)
            
        mode.change(fn=update_visibility, inputs=mode, outputs=[k_slider, threshold_slider])        
        @gr.render(inputs=[model_choices, query, mode, k_slider, threshold_slider], triggers=[submit_button.click])
        def show(model_choices, query, mode, k_slider, threshold_slider):
            num_models = len(model_choices)
            if num_models == 0:
                return gr.Markdown("Please select at least one model to display results.")
            col_num = 6
            min_col = 2
            columns_per_model = max(col_num // num_models, min_col)
            
            with gr.Row():
                if mode == "k":
                    threshold_slider = None 
                if mode == "threshold":
                    k_slider = None
                result_areas = []
                for choice in model_choices:
                    results = search(choice, query, k_slider, threshold_slider)
                    with gr.Column():
                        result_areas.append(
                            gr.Column([gr.Markdown(f"\n# {choice} results\n"),
                            gr.Gallery(value=results, columns=min(columns_per_model, len(results)), label=f"Results for {choice}")]))
                    
                return gr.Row(result_areas)


# Run the interface
demo.launch()

