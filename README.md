# VLM-Based Image Search

This project was created for the Advanced Foundation Models course. It implements an image search functionality using Vision-Language Models (VLM).

## Project Structure

The project is organized as follows:

```plaintext
.
├── data        # Keeps data files
├── logs        # Keeps log files
├── models      # Keeps model caches
├── outputs     # Keeps saved outputs from (SLURM) experiments
├── README.md
└── src         # Contains all code
    ├── clip_search  # CLIP related methods/scripts
    ├── data         # Data related methods/scripts
    ├── demo         # Demo resides here!
    ├── eval         # Evaluation methods/scripts
    ├── gi2t         # GIT related methods/scripts
    ├── services     # FAISS, embedding creation, settings etc.
    └── vlm          # LLaVa related methods/scripts

## Demo
- src/demo: contains previously created captions+embeddings
- src/demo/app_v2.py: Latest demo version, should work as it is

## Search
- src/services/search.py: Contains search and retrieval logic (top k, threshold)

## LLaVa important files
- src/vlm/create_captions.py: Creates captions with LLaVa, requires SLURM
- src/eval/llava_eval.py: Receives predicted captions (json -> filename, caption) and calculates a large table containing all base experiments

## GIT important files
Note: The files are explained in the order one should follow to run experiments
- coco_processor.py: Creates embeddings for COCO images in desired size (also saves IDs), sample command given inside the file.
- git_create_caption.py: Create and save captions for the COCO images given by saved ID file (by coco_processor.py)
- make_response.ipynb : Creates and saves the dictionary (filename:caption) from saved captions, to later use in demo
To reproduce experiment results (or see output cells) with the fixed query set:
- coco_experiment_all.ipynb : Individual experiment with the aggregation methods (avg embed., avg. score, highest score)
- coco_experiment_index.ipynb : Common experiment with the aggregation methods (first, random, concat)
## CLIP important files
- evaluate_multimodal.py: contains evaluation of multimodal search
- generate_images.py: images are generated using stable diffusion
- pseudo_images.py: create embeddings for generated images
- vg_objects.py: evaluate retrieving detected objects in visual genome dataset
