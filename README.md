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
    ├── gi2t         # CLIP related methods/scripts
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
