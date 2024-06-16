# This script generates embeddings for COCO validation dataset captions.
# METHOD:
# --average : If this flag is set, the embeddings of multiple captions for an image are averaged into 1 embedding.
# if not set, each caption is embedded separately and stored with the structure: [image_count, caption_count, embedding_size]  e.g (500,5,384)

# sample command:
# python coco_processor.py --coco_data_dir data/coco-images --size 500 --average --output_dir data/coco_embeddings --save_ids

import torch
from sentence_transformers import SentenceTransformer
from pycocotools.coco import COCO
import argparse
import os
import numpy as np 

def get_image_captions(coco, image_id):
    annIds = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(annIds)
    captions = [ann['caption'] for ann in anns]
    return captions

def embed_text(text,embedder):
    return embedder.encode([text])

def main(args):
    ann_file = os.path.join(args.coco_data_dir, 'annotations/captions_val2017.json')
    coco = COCO(ann_file)
    # get first K images to process
    image_ids = coco.getImgIds()[:args.size]
    
    # set the embedder model for captions
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = []

    for image_id in image_ids:
        captions = get_image_captions(coco, image_id)[:5] # Image with id 96493 has 6 captions, skipped extra caption just to keep the array dim simpler
            
        if args.average:
            caption_embeddings = [embed_text(caption, embedder) for caption in captions]
            caption_embeddings = np.mean(caption_embeddings, axis=0).squeeze()
            embeddings.append(caption_embeddings)
            
        else:
            #for caption in captions:
                #caption_embedding = embed_text(caption, embedder).squeeze()
                #embeddings.append(caption_embedding)
            # group all five captions for an image together in an array and append to the embeddings
            caption_embeddings = [embed_text(caption, embedder) for caption in captions]
            caption_embeddings = np.array(caption_embeddings).squeeze()
            embeddings.append(caption_embeddings)
            

             
    # set the save filename according to the average flag
    if args.average:
        save_filename = os.path.join(args.output_dir, 'coco_embeddings_' + str(args.size) +'_averaged.npy')
    else:
        save_filename = os.path.join(args.output_dir, 'coco_embeddings_' + str(args.size) +'.npy')
        
    
    np.save(save_filename, embeddings)
    
    # save the ids of the images in a separate txt
    if args.save_ids:
        with open(os.path.join(args.output_dir, 'coco_image_ids_' + str(args.size) + '.txt'), 'w') as f:
            for image_id in image_ids:
                f.write(str(image_id) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate embeddings for COCO dataset images and captions')
    parser.add_argument('--coco_data_dir', type=str, default='../../data/coco-images', help='Path to COCO annotation file')
    parser.add_argument('--size', type=int, default=100, help='Number of images to process')
    parser.add_argument('--average', action='store_true', help='Whether to average the embeddings of multiple captions for an image')
    parser.add_argument('--output_dir', type=str, default='../../data/coco_embeddings', help='Path to save the embeddings')
    parser.add_argument('--save_ids', action='store_true', help='Whether to save the ids of the images in a separate file')
    args = parser.parse_args()
    main(args)
