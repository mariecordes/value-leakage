"""Create executive-summary versions of three report figures.

Each output preserves the original plot and adds a compact caption band beneath
it, so the figure can be pasted into the executive summary without a separate
caption block in Notion.

Run with:
    uv run python analysis/make_exec_summary_figures.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent

FIGURES = [
    (
        ROOT / "analysis/target_visibility/lineup_qwen3.5-122b-a10b.png",
        ROOT / "analysis/target_visibility/lineup_qwen3.5-122b-a10b_exec_summary.png",
        "Figure 4. (see Section 3.2)",
        "Qwen's trajectories separate early when the threshold is visible or the model "
        "is instructed to aim; withholding the threshold weakens the shift. The n values "
        "count traces with usable extracted trajectories, while the shift labels use all "
        "usable final answers (n=193 for bet (threshold shown) and n=60 for the other two)." 
        "Figure 3 reports those in more detail.",
    ),
    (
        ROOT / "analysis/continuations/stated_vs_resampled_scatter_qwen3.5-122b-a10b.png",
        ROOT / "analysis/continuations/stated_vs_resampled_scatter_qwen3.5-122b-a10b_exec_summary.png",
        "Figure 6. (see Section 3.3)",
        "Each point compares the estimate Qwen states when stopped with the median of six "
        "continuations from the same prefix. Most points follow the equality line, showing "
        "local predictive consistency.",
    ),
    (
        ROOT / "analysis/cause/cause_consistency.png",
        ROOT / "analysis/cause/cause_consistency_exec_summary.png",
        "Figure 10. (see Section 3.4) ",
        "Every trace shows a positive relationship between the inserted estimate and final "
        "answer, but retention remains far below one. The first spots-per-giraffe estimate "
        "moves the answer consistently while transmitting only a limited share of the "
        "inserted change.",
    ),
]


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    """Use Matplotlib's bundled DejaVu fonts for portable, repeatable output."""
    import matplotlib

    font_dir = Path(matplotlib.get_data_path()) / "fonts/ttf"
    return ImageFont.truetype(str(font_dir / name), size=size)


def wrap(draw: ImageDraw.ImageDraw, text: str, face: ImageFont.FreeTypeFont,
         max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = word if not current else f"{current} {word}"
        if draw.textlength(candidate, font=face) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def add_caption(source: Path, output: Path, label: str, caption: str) -> None:
    image = Image.open(source).convert("RGB")
    width, height = image.size
    body_size = max(22, min(30, round(width * 0.016)))
    label_face = font("DejaVuSans-Bold.ttf", body_size)
    body_face = font("DejaVuSans.ttf", body_size)
    margin_x = round(width * 0.052)
    max_width = width - 2 * margin_x

    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines = wrap(measure, caption, body_face, max_width)
    line_height = round(body_size * 1.38)
    top_pad = round(body_size * 0.85)
    label_height = round(body_size * 1.25)
    body_gap = round(body_size * 0.22)
    bottom_pad = round(body_size * 0.95)
    band_height = top_pad + label_height + body_gap + len(lines) * line_height + bottom_pad

    canvas = Image.new("RGB", (width, height + band_height), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, height, width, height + band_height), fill="#F7F7F5")
    draw.line((margin_x, height, width - margin_x, height), fill="#D8D8D4", width=2)

    y = height + top_pad
    draw.text((margin_x, y), label, font=label_face, fill="#242424")
    y += label_height + body_gap
    for line in lines:
        draw.text((margin_x, y), line, font=body_face, fill="#4A4A46")
        y += line_height

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    print(f"saved {output.relative_to(ROOT)} ({canvas.width}x{canvas.height})")


def main() -> None:
    for source, output, label, caption in FIGURES:
        add_caption(source, output, label, caption)


if __name__ == "__main__":
    main()
