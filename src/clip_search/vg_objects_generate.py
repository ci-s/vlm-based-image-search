import sys
sys.path.append("..")
sys.path.append(".")
from services.data_utilizer import DataUtilizer
from services.vlm_models_service import SearchModel
from eval.evaluate_utils import evaluate_objects
from services.settings import settings
import os
import pandas as pd 
import torch
import spacy 
from tqdm import tqdm 

nlp = spacy.load('en_core_web_sm')
data_util = DataUtilizer("")
data = data_util.load_json_file("../../data/visual_genome/filenames_and_objects.json")
unique_objects = set()
for objects in data.values():
    unique_objects.update(objects)
    
unique_objects = sorted(unique_objects) 
for item in ["a", "an", "ac"]:
    unique_objects.remove(item)
    
def find_roots(phrases):
    # Create a dictionary to store the root forms of the phrases
    root_forms = {}
    for phrase in phrases:
        doc = nlp(phrase)
        # Extract the root token of the phrase
        root_token = [token.lemma_ for token in doc if token.dep_ == 'ROOT']
        
        if root_token:
            root = root_token[0]
            # root = nltk_lemmatizer.lemmatize(root)
            if root not in root_forms:
                root_forms[root] = []
            root_forms[root].append(phrase)
    return root_forms

roots = find_roots(list(unique_objects))
adjusted_roots = {}
for root, forms in roots.items():
    if len(forms) == 1:
        adjusted_roots[forms[0]] = forms
    else:
        adjusted_roots[root] = forms
object_presence = {obj:[] for obj in adjusted_roots.keys()}
for root, forms in adjusted_roots.items():
    for _, objects_list in data.items():
        if len(objects_list) != 0:
            presence = any(form in objects_list for form in forms)
            object_presence[root].append(int(presence))
print(len(object_presence))
        
objects_list = list(unique_objects)
model_names = ["CLIP", "Llava", "GIT"]
output_dir = settings.output_dir
emb_dict = {"CLIP": [], "Llava":[], "GIT":[]}
remove_list = [i for i, (key, value) in enumerate(data.items()) if len(value) == 0]
  
for model_n in model_names:
    embd = data_util.load_texts_embeddings(os.path.join(output_dir, "visual_genome_" + model_n+ "_embeddings.pth"))
    emb_dict[model_n] = torch.stack([tensor for idx, tensor in enumerate(embd) if idx not in remove_list])


for model_n in tqdm(model_names):
    model = SearchModel(model_n).get_model()
    text_embed = model.encode_text_list(list(adjusted_roots.keys()))
    index = data_util.create_index(emb_dict[model_n])
    eval = evaluate_objects(index, text_embed, torch.tensor(list(object_presence.values())))
    df = pd.DataFrame(eval)
    print(df)
    df.to_csv(os.path.join(output_dir, model_n + "_transposed_visual_genome_results_3.csv"))
    df = df.transpose()
    df.to_csv(os.path.join(output_dir, model_n + "_visual_genome_results_3.csv"))



    
    
    

