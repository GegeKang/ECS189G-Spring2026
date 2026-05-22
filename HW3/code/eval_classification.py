import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from dataset import ImagenetteDataset

# 19 diverse templates from the CLIP paper (Radford et al. 2021)
ENSEMBLE_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
    "a photo of a large {}.",
    "a photo of a {} in nature.",
    "there is a {} in the scene.",
    "a bright photo of a {}.",
    "a dark photo of a {}.",
    "a close-up photo of a {}.",
    "a cropped photo of a {}.",
    "a photo of the {}.",
    "a rendition of a {}.",
    "a {} in the wild.",
    "an image of a {}.",
]


def build_arg_parser():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP zero-shot classification on Imagenette validation images."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=project_dir / "clip",
        help="Local CLIP model directory created by download_clip.py.",
    )
    parser.add_argument(
        "--val-txt",
        type=Path,
        default=project_dir / "DATA" / "imagenette2" / "val.txt",
        help="Imagenette validation txt file.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=project_dir / "DATA" / "imagenette2",
        help="Root directory joined with paths from val.txt.",
    )
    parser.add_argument(
        "--prompt",
        choices=["bare", "template", "fancy"],
        default="template",
        help="Prompt style: bare names, single template, or ensemble of 19 templates.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of images to process per CLIP forward pass.",
    )
    return parser


def _extract_tensor(features):
    """Extract raw tensor from HuggingFace model output if needed."""
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    if hasattr(features, "last_hidden_state"):
        return features.last_hidden_state[:, 0]
    return features


@torch.no_grad()
def build_text_features(labels, prompt_style, model, processor, device):
    """Encode class labels into normalized text feature vectors."""
    if prompt_style == "fancy":
        # Encode all templates per class, then average and re-normalize
        class_features = []
        for label in tqdm(labels, desc="Building ensemble text features"):
            prompts = [t.format(label) for t in ENSEMBLE_TEMPLATES]
            inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
            features = F.normalize(_extract_tensor(model.get_text_features(**inputs)), dim=-1)
            # average over templates, then re-normalize
            class_features.append(F.normalize(features.mean(dim=0), dim=-1))
        return torch.stack(class_features)  # [num_classes, D]
    else:
        if prompt_style == "bare":
            prompts = labels
        else:
            prompts = [f"a photo of a {label}" for label in labels]
        inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        features = _extract_tensor(model.get_text_features(**inputs))
        return F.normalize(features, dim=-1)  # [num_classes, D]


@torch.no_grad()
def evaluate(model, processor, dataset, class_labels, text_features, batch_size, device):
    correct = 0
    total = 0

    for start in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        images = [item[0] for item in batch]
        true_labels = [item[1] for item in batch]

        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        image_features = F.normalize(
            _extract_tensor(model.get_image_features(**inputs)), dim=-1
        )  # [B, D]

        similarities = image_features @ text_features.T  # [B, num_classes]
        pred_indices = similarities.argmax(dim=1).cpu().tolist()
        pred_labels = [class_labels[i] for i in pred_indices]

        correct += sum(pred == true for pred, true in zip(pred_labels, true_labels))
        total += len(true_labels)

    return correct / total


def main():
    args = build_arg_parser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ImagenetteDataset(str(args.val_txt), str(args.image_root))
    class_labels = list(dataset.id2label.values())

    model = CLIPModel.from_pretrained(args.model_dir).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_dir)
    model.eval()

    text_features = build_text_features(class_labels, args.prompt, model, processor, device)

    accuracy = evaluate(
        model=model,
        processor=processor,
        dataset=dataset,
        class_labels=class_labels,
        text_features=text_features,
        batch_size=args.batch_size,
        device=device,
    )

    print(f"Prompt style: {args.prompt}")
    print(f"Validation images: {len(dataset)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")


if __name__ == "__main__":
    main()
