from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


DEFAULT_ROOT = Path(
    r"C:\Users\tomar\source\repos\technical-blog\content\posts"
)


# Match a complete HTML <figure>...</figure> block.
FIGURE_RE = re.compile(
    r"<figure\b[^>]*>(.*?)</figure>",
    re.IGNORECASE | re.DOTALL,
)

# Match the <img ...> element inside the figure.
IMG_RE = re.compile(
    r"<img\b(?P<attrs>[^>]*)>",
    re.IGNORECASE | re.DOTALL,
)

# Match an HTML attribute such as src="..." or alt="...".
ATTRIBUTE_RE_TEMPLATE = r'\b{attribute}\s*=\s*"(?P<value>.*?)"'

# Match the visible caption.
FIGCAPTION_RE = re.compile(
    r"<figcaption\b[^>]*>(?P<caption>.*?)</figcaption>",
    re.IGNORECASE | re.DOTALL,
)

# Remove an optional <em> wrapper from the caption.
EM_RE = re.compile(
    r"^\s*<em>(?P<text>.*?)</em>\s*$",
    re.IGNORECASE | re.DOTALL,
)


def get_attribute(attrs: str, attribute: str) -> str | None:
    """Return an HTML attribute value.

    Args:
        attrs: Raw contents of an HTML element's attribute list.
        attribute: Attribute name to retrieve.

    Returns:
        The attribute value, or None if the attribute is not present.
    """
    pattern = re.compile(
        ATTRIBUTE_RE_TEMPLATE.format(attribute=re.escape(attribute)),
        re.IGNORECASE | re.DOTALL,
    )

    match = pattern.search(attrs)

    if not match:
        return None

    return match.group("value").strip()


def clean_caption(caption: str) -> str:
    """Clean the HTML caption for use in the shortcode.

    Args:
        caption: Raw figcaption contents.

    Returns:
        Clean caption text.
    """
    caption = caption.strip()

    # Remove the <em> wrapper used by the existing figure syntax...
    em_match = EM_RE.match(caption)

    if em_match:
        caption = em_match.group("text").strip()

    # Collapse formatting whitespace into normal spaces...
    caption = re.sub(r"\s+", " ", caption)

    return caption


def escape_shortcode_value(value: str) -> str:
    """Escape a value for use inside a Hugo shortcode string.

    Args:
        value: Original string.

    Returns:
        Escaped shortcode string.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


def make_label(src: str) -> str:
    """Generate an accessibility label from an image filename.

    Args:
        src: Image source filename.

    Returns:
        Human-readable lightbox label.
    """
    stem = Path(src).stem
    description = stem.replace("-", " ").replace("_", " ")

    return f"Open full-size {description}"


def convert_figure(match: re.Match[str]) -> str:
    """Convert one HTML figure into a Hugo lightbox shortcode.

    Args:
        match: Regular-expression match containing the complete figure.

    Returns:
        The converted shortcode, or the original figure if it cannot
        safely be converted.
    """
    original = match.group(0)
    contents = match.group(1)

    # Find the image...
    img_match = IMG_RE.search(contents)

    if not img_match:
        return original

    attrs = img_match.group("attrs")

    src = get_attribute(attrs, "src")
    alt = get_attribute(attrs, "alt")

    # Do not convert figures unless we have the minimum information needed...
    if not src or alt is None:
        return original

    # Hugo Page.Resources.GetMatch expects the resource name without "./"...
    if src.startswith("./"):
        src = src[2:]

    # Extract the optional caption...
    caption_match = FIGCAPTION_RE.search(contents)

    caption = None

    if caption_match:
        caption = clean_caption(caption_match.group("caption"))

    label = make_label(src)

    lines = [
        "{{< lightbox",
        f'    src="{escape_shortcode_value(src)}"',
        f'    alt="{escape_shortcode_value(alt)}"',
        f'    label="{escape_shortcode_value(label)}"',
    ]

    if caption:
        lines.append(
            f'    caption="{escape_shortcode_value(caption)}"'
        )

    lines.append(">}}")

    return "\n".join(lines)


def process_file(path: Path, write: bool) -> int:
    """Convert figure blocks in one Markdown file.

    Args:
        path: Path to index.md.
        write: Whether changes should actually be written.

    Returns:
        Number of figures converted.
    """
    original_text = path.read_text(encoding="utf-8")

    # Do not accidentally process shortcode markup as HTML...
    converted_text, count = FIGURE_RE.subn(
        convert_figure,
        original_text,
    )

    # subn() counts every regex match, including figures that convert_figure()
    # deliberately leaves unchanged. Calculate the actual number changed...
    original_figures = FIGURE_RE.findall(original_text)
    converted_figures = FIGURE_RE.findall(converted_text)

    # A simple and reliable way to calculate successful conversions is to
    # compare lightbox shortcode counts before and after...
    before = original_text.count("{{< lightbox")
    after = converted_text.count("{{< lightbox")
    converted_count = after - before

    if converted_count == 0:
        return 0

    print(f"{path}")
    print(f"    {converted_count} figure(s) converted")

    if write:
        backup = path.with_name("index.md.bak")

        # Preserve the original file before modifying it...
        if not backup.exists():
            shutil.copy2(path, backup)

        path.write_text(
            converted_text,
            encoding="utf-8",
            newline="",
        )

    return converted_count


def main() -> None:
    """Convert article figures to Signal & Syntax lightbox shortcodes."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert HTML figure blocks in Hugo index.md files "
            "to lightbox shortcodes."
        )
    )

    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root posts directory.",
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help=(
            "Actually modify files. Without this option the script "
            "performs a dry run."
        ),
    )

    args = parser.parse_args()

    root = args.root

    if not root.exists():
        raise SystemExit(f"Directory does not exist: {root}")

    mode = "WRITE" if args.write else "DRY RUN"

    print(f"Mode: {mode}")
    print(f"Root: {root}")
    print()

    files_changed = 0
    figures_changed = 0

    # Recursively process index.md files only...
    for path in sorted(root.rglob("index.md")):
        count = process_file(path, args.write)

        if count:
            files_changed += 1
            figures_changed += count

    print()
    print(f"Files affected:   {files_changed}")
    print(f"Figures affected: {figures_changed}")

    if not args.write:
        print()
        print("No files were changed.")
        print("Run again with --write to apply the conversions.")


if __name__ == "__main__":
    main()