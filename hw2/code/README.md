# ECS 189G Spring 2026 - Assignment 2: Prompting with In-Context Learning and Evaluation

In this assignment, you will explore the power and limitations of In-Context Learning (ICL). You will design prompts to guide a base Large Language Model (`Qwen3-0.6B-Base`) to perform sentiment analysis on the SST-2 dataset. You will also rigorously evaluate how the number of exemplars and prompt formatting affect the model's performance.

## File Structure & Provided Code

- `DATA/SST2/`: Directory containing the `train.json` and `val.json` dataset files.
- `dataset.py`: Script to load and explore the SST-2 dataset format.
- `download_qwen.py`: Script to download the pre-trained Qwen3-0.6B-Base model to your local machine.
- `qwen.py`: A starter script demonstrating a basic 4-shot In-Context Learning setup.
- `requirements.txt`: Python package dependencies for this assignment.
  
## Environment Setup

It is highly recommended to use a virtual environment (e.g., Conda or Python venv) to avoid dependency conflicts.

1. Ensure you have Python 3.8+ installed.
2. Install the required dependencies:
   ```bash
   conda create -n in_context_learning python=3.10.11
   conda activate in_context_learning
   cd /path/to/this/folder
   pip install -r requirements.txt
   ```
(Note: The model `Qwen3-0.6B-Base` is CPU-friendly and will automatically use a GPU if available.)


## Task Instructions

### Task 1: Complete the Transformer Model

Run the provided dataset exploration script:
```bash
python dataset.py
```
**Your Task**: Observe the printed examples and understand the data structure. Note that the dataset uses numerical labels: 0 represents Negative and 1 represents Positive. When constructing your prompts later, you will need to map these to natural language words (e.g., "Positive" / "Negative").

### Task 2: Download the Pre-trained Model

Run the following script to download the `Qwen3-0.6B-Base` model to your local directory:
```bash
python download_qwen.py # This may take a few minutes depending on your internet connection
```

### Task 3: First Steps with In-Context Learning

Open and run the `qwen.py` script to see In-Context Learning in action:
```bash
python qwen.py
```
**Your Task**: 
1. Observe the model's output based on the provided 4-shot prompt.
2. Open `qwen.py` and manually change the `your_review_here` variable. Try writing your own movie reviews with strong positive or negative sentiments.
3. Record your findings: How does the model's output change based on your input? Does it accurately predict the sentiment? Document these observations in your final report.

### Task 4: Build the Evaluation Pipeline & The "Parsing" Challenge
Create a new Python script (e.g., `eval_sst2.py`) to systematically evaluate the model on the `val.json` validation set.
1. Dynamic Prompting: Write a function to randomly (or selectively) sample exemplars from `train.json` and prepend them as context to the validation queries.

2. The Parsing Challenge: We are using a Base Model, not an instruction-tuned model. It does **not know when to stop generating**. It will continue spitting out tokens until it hits the `max_new_tokens` limit.

**Your Task**: You must implement a parsing logic in your script to extract only the target label ("Positive" or "Negative") from the model's continuous text generation.

### Task 5: K-Shot Scaling Experiment
Using your evaluation script from Task 4, investigate how the number of in-context exemplars ($k$) impacts the classification accuracy on the validation set.

**Your Task**:
1. Run your evaluation pipeline for $k \in \{0, 1, 3, 5, 8, 16\}$ shots.
2. Record the Exact Match Accuracy for each $k$.
3. Analysis: Plot an Accuracy vs. $k$-shot curve. In your report, analyze the results. For example, how does the model perform at $k=0$? How does the model utilize the increasing number of exemplars to "learn" the task?


### Task 6: Prompt Sensitivity & Robustness
Base models are notoriously brittle and highly sensitive to prompt formatting. Fix your exemplar count to a specific number (e.g., $k=3$ or $k=5$) and test different prompt templates.
**Your Task**: Implement and evaluate at least two distinct variants:
1. Format Shift: Change the layout of the prompt. For example, instead of `Review: [Text] \n Sentiment: Positive`, try `Input: [Text] | Label: Positive`. You can design the template yourself.
2. Label Word Shift: Swap the target label words from `Positive/Negative` to `Good/Bad` or `1/0`.
3. Analysis: Compare the accuracy of these variants against your standard prompt. Discuss in your report how base models perform when faced with minor formatting changes.


## Submission Guidelines
Please submit a single .zip file containing:
1. Your complete Python evaluation script(s) (show your dynamic prompt building and text parsing/truncation logic).
2. A PDF Report containing: 
   - Your observations from manually modifying `your_review_here` in Task 3.
   - The Accuracy vs. $k$-shot plot and your analysis for Task 5.
   - The prompt templates you designed for Task 6, their corresponding accuracies, and your analysis.