import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

local_dir = './Qwen3-0.6B-Base'
model = AutoModelForCausalLM.from_pretrained(local_dir, dtype=torch.float32, attn_implementation="eager").to(device)
tokenizer = AutoTokenizer.from_pretrained(local_dir)

LABEL_MAP = {0: 'Negative', 1: 'Positive'}


def load_data(path):
    with open(path) as f:
        return json.load(f)


# Build a k-shot prompt by prepending exemplars before the query.
def build_prompt(exemplars, query_sentence):
    lines = []
    for ex in exemplars:
        lines.append(f"Review: {ex['sentence'].strip()}")
        lines.append(f"Sentiment: {LABEL_MAP[ex['label']]}")
        lines.append("")
    lines.append(f"Review: {query_sentence.strip()}")
    lines.append("Sentiment:")
    return "\n".join(lines)

 
 # Extract the first Positive/Negative token after the prompt.
def parse_prediction(generated_text, prompt):
    # Strip the prompt prefix to get only the new tokens
    completion = generated_text[len(prompt):]
    # Take the first word and normalize
    first_word = completion.strip().split()[0] if completion.strip() else ""
    first_word = first_word.strip(".,!?\"'").capitalize()
    if first_word in ("Positive", "Negative"):
        return first_word
    # Fallback: search completion for either label word
    lower = completion.lower()
    if "positive" in lower:
        return "Positive"
    if "negative" in lower:
        return "Negative"
    return "Unknown"


# Run k-shot ICL evaluation on val_data, sampling num_samples examples."""
def evaluate(train_data, val_data, k=4, num_samples=None, seed=42):
    random.seed(seed)
    samples = val_data if num_samples is None else random.sample(val_data, num_samples)

    correct = 0
    for item in samples:
        # Randomly sample k exemplars from train set
        exemplars = random.sample(train_data, k) if k > 0 else []
        prompt = build_prompt(exemplars, item['sentence'])

        input_ids = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**input_ids, max_new_tokens=10)
        generated = tokenizer.decode(output[0], skip_special_tokens=True)

        pred = parse_prediction(generated, prompt)
        gold = LABEL_MAP[item['label']]
        if pred == gold:
            correct += 1

    accuracy = correct / len(samples)
    print(f"k={k} | Samples={len(samples)} | Accuracy={accuracy:.4f} ({correct}/{len(samples)})")
    return accuracy


if __name__ == "__main__":
    train_data = load_data('./DATA/SST2/train.json')
    val_data = load_data('./DATA/SST2/val.json')

    # Task 4: single run with k=4 on a small subset to verify the pipeline
    evaluate(train_data, val_data, k=4, num_samples=50)
