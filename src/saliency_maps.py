import cv2
import argparse
from pathlib import Path

def generate_saliency(image_path: Path, output_maps: Path, output_overlays: Path) -> None:
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Skipped unreadable image: {image_path}")
        return

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(image)

    if not success:
        print(f"Failed to generate saliency map for: {image_path}")
        return

    saliency_map = (saliency_map * 255).astype("uint8")
    heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    output_maps.mkdir(parents = True, exist_ok = True)
    output_overlays.mkdir(parents = True, exist_ok = True)

    cv2.imwrite(str(output_maps / f"{image_path.stem}_saliency_map.jpg"), saliency_map)
    cv2.imwrite(str(output_overlays / f"{image_path.stem}_saliency_overlay.jpg"), overlay)

    print(f"Generated saliency outputs for: {image_path.name}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required = True)
    parser.add_argument("--maps", required = True)
    parser.add_argument("--overlays", required = True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_maps = Path(args.maps)
    output_overlays = Path(args.overlays)

    valid_extensions = {".jpg", ".jpeg", ".png"}

    for image_path in input_dir.iterdir():
        if image_path.suffix.lower() in valid_extensions:
            generate_saliency(image_path, output_maps, output_overlays)

if __name__ == "__main__":
    main()