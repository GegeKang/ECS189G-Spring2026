import json
import random
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

local_dir = '/Users/gegekang/Desktop/ECS189G-Spring2026/hw2/code/Qwen3-0.6B-Base'
model = AutoModelForCausalLM.from_pretrained(local_dir, dtype=torch.float32, attn_implementation="eager").to(device)
tokenizer = AutoTokenizer.from_pretrained(local_dir)

LABEL_MAP = {0: 'Negative', 1: 'Positive'}


def load_data(path):
    with open(path) as f:
        return json.load(f)


def build_prompt(exemplars, query_sentence):
    lines = []
    for ex in exemplars:
        lines.append(f"Review: {ex['sentence'].strip()}")
        lines.append(f"Sentiment: {LABEL_MAP[ex['label']]}")
        lines.append("")
    lines.append(f"Review: {query_sentence.strip()}")
    lines.append("Sentiment:")
    return "\n".join(lines)


def parse_prediction(generated_text, prompt):
    completion = generated_text[len(prompt):]
    first_word = completion.strip().split()[0] if completion.strip() else ""
    first_word = first_word.strip(".,!?\"'").capitalize()
    if first_word in ("Positive", "Negative"):
        return first_word
    lower = completion.lower()
    if "positive" in lower:
        return "Positive"
    if "negative" in lower:
        return "Negative"
    return "Unknown"


def evaluate_k(train_data, val_data, k, num_samples=50, seed=42):
    random.seed(seed)
    samples = random.sample(val_data, num_samples)

    correct = 0
    for item in samples:
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
    print(f"k={k:>2} | Accuracy={accuracy:.4f} ({correct}/{len(samples)})")
    return accuracy


if __name__ == "__main__":
    train_data = load_data('./DATA/SST2/train.json')
    val_data = load_data('./DATA/SST2/val.json')

    k_values = [0, 1, 3, 5, 8, 16]
    accuracies = []

    for k in k_values:
        acc = evaluate_k(train_data, val_data, k, num_samples=50)
        accuracies.append(acc)

    # Plot accuracy vs k
    plt.figure()
    plt.plot(k_values, accuracies, marker='o')
    plt.xlabel('Number of shots (k)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs K-Shot')
    plt.xticks(k_values)
    plt.grid(True)
    plt.savefig('task5_kshot.png', dpi=150)
    print("Plot saved to task5_kshot.png")
