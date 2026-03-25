
import cv2
import numpy as np
from skimage import filters
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # preprocessing
    gaussian_sigma: float = 1.75       # light smoothing — preserve membrane edges
    clahe_clip: float = 2.0           # CLAHE clip limit (absolute, not ratio)
    clahe_tile: tuple = (8, 8)

    # Canny edge thresholds
    canny_low: int = 20
    canny_high: int = 60

    # Contour filters — tuned to zebrafish edema sac size at typical magnification
    min_area: int = 2_000             
    max_area: int = 120_000        
    min_solidity: float = 0.82        
    max_aspect_ratio: float = 2.2    
    min_mean_intensity: int = 140     


def load_and_enhance(image_path: str, cfg: PipelineConfig):
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot open: {image_path}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE to normalise illumination variation across the specimen
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=cfg.clahe_tile)
    equalized = clahe.apply(gray)

    # gaussian blur to reduce noise before edge detection
    gray_f = equalized.astype(np.float32) / 255.0
    smoothed = img_as_ubyte(filters.gaussian(gray_f, sigma=cfg.gaussian_sigma))

    return bgr, gray, smoothed


def detect_edema_contours(bgr, gray, smoothed, cfg: PipelineConfig):
    H, W = gray.shape

    #canny edge detection to find potential edema sac boundaries
    edges = cv2.Canny(smoothed, cfg.canny_low, cfg.canny_high)

    # dilate + close to connect fragmented edges and fill small gaps
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)

    # find contours from the processed edge map
    contours, _ = cv2.findContours(
        edges_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    # score
    print(f"\nTotal contours found: {len(contours)}")
    print(f"{'idx':>4}  {'area':>8}  {'solidity':>8}  {'AR':>5}  "
          f"{'mean_int':>8}  {'verdict'}")
    print("-" * 55)

    candidates = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if not (cfg.min_area < area < cfg.max_area):
            continue

        # solidity = area / convex hull area to filter out irregular shapes 
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # aspect ratio from bounding rect
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / (min(w, h) + 1e-5)

        # mean intensity within the contour 
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
        mean_int = cv2.mean(gray, mask=mask)[0]

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        passed = (
            solidity >= cfg.min_solidity
            and aspect <= cfg.max_aspect_ratio
            and mean_int >= cfg.min_mean_intensity
        )

        reason = []
        if solidity < cfg.min_solidity:
            reason.append(f"solidity={solidity:.2f}")
        if aspect > cfg.max_aspect_ratio:
            reason.append(f"AR={aspect:.1f}")
        if mean_int < cfg.min_mean_intensity:
            reason.append(f"intensity={mean_int:.0f}")

        verdict = "✓ PASS" if passed else f"✗ {', '.join(reason)}"
        print(f"{i:>4}  {area:>8.0f}  {solidity:>8.2f}  {aspect:>5.1f}  "
              f"{mean_int:>8.1f}  {verdict}")

        if passed:
            candidates.append({
                "contour": cnt,
                "area": area,
                "solidity": solidity,
                "aspect": aspect,
                "mean_intensity": mean_int,
                "centroid": (cx, cy),
                "bbox": (x, y, w, h),
            })

    # result
    result = {"YE": None, "PE": None}
    if not candidates:
        print("\n⚠️  No candidates passed all filters.")
        print("    Try lowering min_solidity, min_area, or min_mean_intensity.")
        return result, edges_closed

    # yolk edema
    candidates.sort(key=lambda c: c["area"], reverse=True)
    result["YE"] = candidates[0]
    print(f"\n→ YE: area={result['YE']['area']:.0f}  "
          f"solidity={result['YE']['solidity']:.2f}  "
          f"centroid={result['YE']['centroid']}")

    # cardiac edema
    if len(candidates) > 1:
        rest = sorted(candidates[1:], key=lambda c: c["centroid"][0], reverse=True)
        result["PE"] = rest[0]
        print(f"→ PE: area={result['PE']['area']:.0f}  "
              f"solidity={result['PE']['solidity']:.2f}  "
              f"centroid={result['PE']['centroid']}")

    return result, edges_closed


def visualize(bgr, edges, result):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0d0d0d")
    titles = ["1. Original", "2. Canny Edges", "3. Final: YE + PE"]
    imgs = [
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        edges,
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
    ]

    for ax, title, img in zip(axes, titles, imgs):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title, color="white", fontsize=11)
        ax.axis("off")

    color_map = {"YE": (0, 217, 255), "PE": (255, 60, 60)}
    ax_out = axes[2]

    for name, info in result.items():
        if info is None:
            continue
        # draw filled contour with transparency
        overlay = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
        c = tuple(int(v) for v in color_map[name])
        cv2.drawContours(overlay, [info["contour"]], -1, c, thickness=cv2.FILLED)
        alpha = 0.25
        base = ax_out.get_images()
        if base:
            cur = (base[0].get_array() * 255).astype(np.uint8)
            blended = cv2.addWeighted(cur, 1 - alpha, overlay, alpha, 0)
            base[0].set_data(blended)

        # outline
        pts = info["contour"][:, 0, :]
        ax_out.plot(
            np.append(pts[:, 0], pts[0, 0]),
            np.append(pts[:, 1], pts[0, 1]),
            color=[v / 255 for v in color_map[name]], linewidth=2.5
        )
        cx, cy = info["centroid"]
        ax_out.text(
            cx, cy - 15, name,
            color=[v / 255 for v in color_map[name]],
            fontsize=13, weight="bold", ha="center",
            bbox=dict(facecolor="black", alpha=0.65, edgecolor="none", pad=3)
        )

    plt.tight_layout()
    plt.savefig("zebrafish_edemas.png", dpi=150, facecolor="#0d0d0d")
    plt.show()


def run_pipeline(image_path: str):
    cfg = PipelineConfig()
    bgr, gray, smoothed = load_and_enhance(image_path, cfg)
    result, edges = detect_edema_contours(bgr, gray, smoothed, cfg)
    visualize(bgr, edges, result)


if __name__ == "__main__":
    run_pipeline("data/image3.jpg")