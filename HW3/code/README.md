# ECS 189G Spring 2026 - Assignment 3: Multimodal Foundation Models and Zero-Shot Alignment

In this assignment, you will explore the powerful capabilities of **CLIP (Contrastive Language-Image Pre-training)** in a strictly training-free (zero-shot) setting. You will evaluate its performance on image classification and build a cross-modal text-to-image retrieval system from scratch.

## File Structure & Provided Code

- `DATA/imagenette2/`: Directory containing the compressed dataset `imagenette2-160.tgz` and the metadata file `val.txt`.
- `clip.py`: A basic inference script demonstrating how to load CLIP and compute image-text similarity probabilities.
- `dataset.py`: Contains the `ImagenetteDataset` PyTorch class to load and format the image data.
- `download_clip.py`: Script to download the pre-trained `openai/clip-vit-base-patch32` model locally.
- `requirements.txt`: Python package dependencies.
  

## Environment Setup

It is highly recommended to use a virtual environment (e.g., Conda or Python venv) to avoid dependency conflicts.

1. Ensure you have Python 3.8+ installed.
2. Install the required dependencies:
   ```bash
   conda create -n clip python=3.10.11
   conda activate clip
   cd /path/to/ECS189G/HW3
   pip install -r requirements.txt
   ```
(Note: The model `openai/clip-vit-base-patch32` is CPU-friendly and will automatically use a GPU if available.)

## Task Instructions

### Task 1: Download Model & Understand Basic Inference

First, download the pre-trained CLIP model to your local machine:
```bash
python3 download_clip.py
```
Once the download is complete, run the provided `clip.py` script:
```bash
python3 clip.py
```
**Your Task**: Review the `clip.py` code. Observe how it processes an image and a list of text strings, passes them through the model, and computes the Softmax probabilities based on the image-text similarity scores (`logits_per_image`).


### Task 2: Prepare and Explore the Dataset

We will use the Imagenette dataset (a 10-class subset of ImageNet) for zero-shot classification.
First, extract the compressed images:
```bash
tar zxvf DATA/imagenette2/imagenette2-160.tgz -C DATA/imagenette2/
```
Next, run the dataset script to perform a sanity check and view the data structure:
```bash
python3 dataset.py
```
**Your Task**: Understand how the `ImagenetteDataset` class maps image paths to their corresponding labels (e.g., "tench", "parachute").

### Task 3: Zero-Shot Image Classification
Create a new Python script (e.g., `eval_classification.py`).

**Your Task**: 
1. Instantiate the `ImagenetteDataset` using the provided validation text file.
2. Iterate through the dataset and use the `CLIPProcessor` and `CLIPModel` to perform zero-shot classification.
3. For each image, calculate its similarity score against all 10 possible class labels defined in `dataset.id2label`.
4. Predict the class with the highest probability (using `argmax`) and calculate the overall **Accuracy** on the entire validation set.

### Task 4: Visual Prompt Engineering
CLIP is sensitive to how text prompts are formatted.

**Your Task**: Modify the text prompts in your classification script and observe how the accuracy changes.
- Baseline: Use just the bare class name (e.g., `["tench", "church", ...]`).
- Templated Prompt: Wrap the class name in a descriptive sentence (e.g., `["a photo of a tench", "a photo of a church", ...]`).
- Analysis: Record the accuracies for both approaches in your report. Discuss why providing contextual language (templates) improves or degrades the performance of a vision-language model.

### Task 5: Text-to-Image Retrieval System
Now you will build a mini search engine. We will use a larger dataset (COCO validation 2017, ~5,000 images, 1GB) for this task.
1. Download COCO: Download and extract the dataset to a new directory (e.g., `DATA/coco/`):
```bash
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d DATA/coco/
```
2. Build the Retrieval System: Create a new script (e.g., `retrieval.py`).
- Offline Phase: Iterate through the ~5,000 COCO images, extract their image embeddings using CLIP, and store them in a large Tensor.
- Online Phase: Write a function that takes a complex, descriptive text query (e.g., "a dog catching a frisbee in the air"), computes the cosine similarity against all stored image embeddings, and retrieves the top 5 most similar images.
3. Visualization: Use matplotlib (or a similar library) to plot and save the top 5 retrieved images for at least 3 distinct, creative text queries.

## Submission Guidelines
Please submit a single .zip file containing:
1. Your Python scripts (e.g., `eval_classification.py`, `retrieval.py`.).
2. A PDF Report containing: 
   - The overall accuracy of your zero-shot classification (Task 3).
   - Your Prompt Engineering accuracy comparison and analysis (Task 4).
   - The output images showing your top-5 retrieval results from Task 5. A qualitative analysis of your retrieval system (Task 5). Did the model retrieve accurate images? Where did it fail or retrieve conceptually similar but incorrect images?
