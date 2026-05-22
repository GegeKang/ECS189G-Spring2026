import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

local_dir = './Qwen3-0.6B-Base'
model = AutoModelForCausalLM.from_pretrained(local_dir, dtype=torch.float32, attn_implementation="eager").to(device)
tokenizer = AutoTokenizer.from_pretrained(local_dir)

K = 5
NUM_SAMPLES = 50
SEED = 42


def load_data(path):
    with open(path) as f:
        return json.load(f)


# --- Prompt builders ---

def build_standard(exemplars, query):
    """Baseline: Review/Sentiment format with Positive/Negative labels."""
    label_map = {0: 'Negative', 1: 'Positive'}
    lines = []
    for ex in exemplars:
        lines.append(f"Review: {ex['sentence'].strip()}")
        lines.append(f"Sentiment: {label_map[ex['label']]}")
        lines.append("")
    lines.append(f"Review: {query.strip()}")
    lines.append("Sentiment:")
    return "\n".join(lines)


def build_format_shift(exemplars, query):
    """Variant 1 (Format Shift): pipe-separated Input/Label layout."""
    label_map = {0: 'Negative', 1: 'Positive'}
    lines = []
    for ex in exemplars:
        lines.append(f"Input: {ex['sentence'].strip()} | Label: {label_map[ex['label']]}")
    lines.append(f"Input: {query.strip()} | Label:")
    return "\n".join(lines)


def build_label_shift(exemplars, query):
    """Variant 2 (Label Word Shift): same format but Good/Bad instead of Positive/Negative."""
    label_map = {0: 'Bad', 1: 'Good'}
    lines = []
    for ex in exemplars:
        lines.append(f"Review: {ex['sentence'].strip()}")
        lines.append(f"Sentiment: {label_map[ex['label']]}")
        lines.append("")
    lines.append(f"Review: {query.strip()}")
    lines.append("Sentiment:")
    return "\n".join(lines)


# --- Parsers ---

def parse_pos_neg(generated, prompt):
    completion = generated[len(prompt):]
    first_word = completion.strip().split()[0] if completion.strip() else ""
    first_word = first_word.strip(".,!?\"'|").capitalize()
    if first_word in ("Positive", "Negative"):
        return first_word
    lower = completion.lower()
    if "positive" in lower:
        return "Positive"
    if "negative" in lower:
        return "Negative"
    return "Unknown"


def parse_format_shift(generated, prompt):
    # Completion follows "| Label:" — look for Positive/Negative
    completion = generated[len(prompt):]
    first_word = completion.strip().split()[0] if completion.strip() else ""
    first_word = first_word.strip(".,!?\"'|").capitalize()
    if first_word in ("Positive", "Negative"):
        return first_word
    lower = completion.lower()
    if "positive" in lower:
        return "Positive"
    if "negative" in lower:
        return "Negative"
    return "Unknown"


def parse_good_bad(generated, prompt):
    completion = generated[len(prompt):]
    first_word = completion.strip().split()[0] if completion.strip() else ""
    first_word = first_word.strip(".,!?\"'").capitalize()
    if first_word == "Good":
        return "Positive"
    if first_word == "Bad":
        return "Negative"
    lower = completion.lower()
    if "good" in lower:
        return "Positive"
    if "bad" in lower:
        return "Negative"
    return "Unknown"


# --- Evaluation ---

def evaluate(train_data, val_data, prompt_fn, parse_fn, label, num_samples=NUM_SAMPLES, seed=SEED):
    random.seed(seed)
    samples = random.sample(val_data, num_samples)
    gold_map = {0: 'Negative', 1: 'Positive'}

    correct = 0
    for item in samples:
        exemplars = random.sample(train_data, K)
        prompt = prompt_fn(exemplars, item['sentence'])

        input_ids = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**input_ids, max_new_tokens=10)
        generated = tokenizer.decode(output[0], skip_special_tokens=True)

        pred = parse_fn(generated, prompt)
        gold = gold_map[item['label']]
        if pred == gold:
            correct += 1

    accuracy = correct / len(samples)
    print(f"{label:<35} | k={K} | Accuracy={accuracy:.4f} ({correct}/{len(samples)})")
    return accuracy


if __name__ == "__main__":
    train_data = load_data('./DATA/SST2/train.json')
    val_data = load_data('./DATA/SST2/val.json')

    results = {}
    results['Standard (Review/Sentiment, Pos/Neg)'] = evaluate(
        train_data, val_data, build_standard, parse_pos_neg,
        'Standard (Review/Sentiment, Pos/Neg)')

    results['Format Shift (Input|Label)'] = evaluate(
        train_data, val_data, build_format_shift, parse_format_shift,
        'Format Shift (Input|Label)')

    results['Label Shift (Good/Bad)'] = evaluate(
        train_data, val_data, build_label_shift, parse_good_bad,
        'Label Shift (Good/Bad)')

    print("\n--- Summary ---")
    for name, acc in results.items():
        print(f"  {name}: {acc:.4f}")
