import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


DEFAULT_QUERIES = [
    "a dog catching a frisbee in the air",
    "a person riding a bicycle on a city street",
    "a plate of food on a dining table",
]


def build_arg_parser():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build a CLIP text-to-image retrieval index for COCO val2017."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=project_dir / "clip",
        help="Local CLIP model directory created by download_clip.py.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=project_dir / "DATA" / "coco" / "val2017",
        help="Directory containing COCO val2017 jpg images.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=project_dir / "DATA" / "coco" / "clip_val2017_embeddings.pt",
        help="Where image paths and normalized CLIP embeddings are cached.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "retrieval_results",
        help="Directory for saved top-5 retrieval visualizations.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of images to encode per CLIP forward pass.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of images to retrieve for each text query.",
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Text query. Pass multiple --query flags to evaluate multiple queries.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Recompute image embeddings even if the cache exists.",
    )
    return parser


def load_image(path):
    return Image.open(path).convert("RGB")


@torch.no_grad()
def encode_images(model, processor, image_paths, batch_size, device):
    embeddings = []

    for start in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[start : start + batch_size]
        images = [load_image(path) for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        image_features = model.get_image_features(**inputs)
        # get_image_features may return a model output object instead of a tensor
        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output
        elif hasattr(image_features, "last_hidden_state"):
            image_features = image_features.last_hidden_state[:, 0]
        image_features = F.normalize(image_features, dim=1)
        embeddings.append(image_features.cpu())

    return torch.cat(embeddings, dim=0)


def load_or_build_index(args, model, processor, device):
    if args.cache_path.exists() and not args.rebuild_cache:
        cache = torch.load(args.cache_path, map_location="cpu")
        return cache["image_paths"], cache["image_embeddings"]

    image_paths = sorted(args.image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(
            f"No .jpg images found in {args.image_dir}. Download and unzip COCO val2017 first."
        )

    image_embeddings = encode_images(
        model=model,
        processor=processor,
        image_paths=image_paths,
        batch_size=args.batch_size,
        device=device,
    )

    args.cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_paths": [str(path) for path in image_paths],
            "image_embeddings": image_embeddings,
        },
        args.cache_path,
    )
    return [str(path) for path in image_paths], image_embeddings


@torch.no_grad()
def retrieve(query, model, processor, image_embeddings, top_k, device):
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    text_features = model.get_text_features(**inputs)
    if hasattr(text_features, "pooler_output"):
        text_features = text_features.pooler_output
    elif hasattr(text_features, "last_hidden_state"):
        text_features = text_features.last_hidden_state[:, 0]
    text_features = F.normalize(text_features, dim=1).cpu()
    similarities = image_embeddings @ text_features.squeeze(0)
    scores, indices = similarities.topk(top_k)
    return scores.tolist(), indices.tolist()


def safe_filename(text):
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in text)
    return "_".join(part for part in cleaned.split("_") if part)[:80]


def save_results(query, result_paths, scores, output_path):
    fig, axes = plt.subplots(1, len(result_paths), figsize=(4 * len(result_paths), 4))
    if len(result_paths) == 1:
        axes = [axes]

    for rank, (axis, image_path, score) in enumerate(zip(axes, result_paths, scores), start=1):
        axis.imshow(load_image(image_path))
        axis.set_title(f"#{rank}\nscore={score:.3f}", fontsize=10)
        axis.axis("off")

    fig.suptitle(query, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    args = build_arg_parser().parse_args()
    queries = args.queries or DEFAULT_QUERIES
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(args.model_dir).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_dir)
    model.eval()

    image_paths, image_embeddings = load_or_build_index(args, model, processor, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Indexed images: {len(image_paths)}")
    for query in queries:
        scores, indices = retrieve(
            query=query,
            model=model,
            processor=processor,
            image_embeddings=image_embeddings,
            top_k=args.top_k,
            device=device,
        )
        result_paths = [image_paths[i] for i in indices]
        output_path = args.output_dir / f"{safe_filename(query)}.png"
        save_results(query, result_paths, scores, output_path)

        print(f"\nQuery: {query}")
        for rank, (path, score) in enumerate(zip(result_paths, scores), start=1):
            print(f"{rank}. {Path(path).name}  score={score:.4f}")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
