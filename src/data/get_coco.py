
from torchvision.datasets import CocoCaptions
from torchvision import transforms
from open_clip import create_model_from_pretrained
model_id = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"

_, clip_processor = create_model_from_pretrained(model_id)

coco_root = "/storage/group/dataset_mirrors/old_common_datasets/coco2017/coco_val17/val2017"
coco_ann_file = "/storage/group/dataset_mirrors/old_common_datasets/coco2017/annotations_trainval2017/annotations/captions_val2017.json"

target_size = {"Llava": (336,336),
               "GIT"  : (224,224),
               "CLIP" : (378,378)}
img_transformer = lambda model_n : transforms.Compose([transforms.Resize(target_size[model_n]),
                                                       transforms.ToTensor()])
def get_cocos_like_dataset(img_transform, text_tokenize, captions_per_img,
                           img_root=coco_root, ann_root=coco_ann_file):
    if text_tokenize is None:
        target_transform = lambda texts : texts[:captions_per_img]
    else:
        target_transform = lambda texts :text_tokenize(texts[:captions_per_img])
    if img_transform is None:
        img_transform = lambda images : transforms.ToTensor()(images)
            
    dataset = CocoCaptions(
        root = img_root,
        annFile = ann_root,
        transform = img_transform,
        target_transform = target_transform
    )
    return dataset


def load_cocos_like_dataset(captions_per_img, model_name, img_root = None, ann_root=None):
    if img_root is None:
        img_root = coco_root
    if ann_root is None:
        ann_root = coco_ann_file
    img_transform = None
    img_transform = get_image_pretransformer(model_name)
    return get_cocos_like_dataset(img_transform, None, captions_per_img, img_root, ann_root)

def get_image_pretransformer(model_name):
    if model_name == "CLIP":
        return clip_processor
    else:
        return img_transformer(model_name)
