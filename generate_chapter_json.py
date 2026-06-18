#!/usr/bin/env python3
"""Generate simple per-chapter JSON files from saved Wisdomlib HTML with English translations and purports."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

from generate_sanskrit_excel import parse_data
from scrape_chapter import extract_analysis_text

HTML_RE = re.compile(r"^(\d+)_(\d+)_(\d+)\.html$")


def devanagari_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    meta = soup.find("meta", attrs={"name": "description"})
    if not meta or not meta.get("content"):
        return ""
    content = re.sub(r"\s+", " ", meta["content"]).strip()
    return re.sub(r"^Verse\s+[\d.]+:\s*", "", content).strip()


def extract_verse_numbers_and_text(text: str):
    """
    Parses verse number references at the beginning of a paragraph.
    Matches formats like '1. ', '1-3. ', '1, 2. ', '1, 2, 3. ', '2. 3. '
    """
    m = re.match(r"^\s*((?:(?:\d+)(?:\s*-\s*\d+)?\s*[\.,]\s*)+)(.*)$", text, re.DOTALL)
    if not m:
        return None, text
        
    prefix_str = m.group(1).strip()
    rest_text = m.group(2).strip()
    
    chunks = re.split(r"[\s,\.]+", prefix_str)
    verses = []
    for chunk in chunks:
        if not chunk:
            continue
        if "-" in chunk:
            parts = chunk.split("-")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                start, end = int(parts[0]), int(parts[1])
                verses.extend(range(start, end + 1))
        elif chunk.isdigit():
            verses.append(int(chunk))
            
    if not verses:
        return None, text
    return verses, rest_text


def parse_english_translation(html_content: str) -> dict:
    """
    Parses the English translation page to extract translations and footnotes/purports.
    Returns a dictionary mapping verse_num (int) -> {'translation': str, 'purport': str}
    """
    soup = BeautifulSoup(html_content, "lxml")
    container = soup.find(class_="chapter-content")
    if not container:
        container = soup.find(class_="chapterDetail")
    if not container:
        container = soup.body
        
    elements = container.find_all(["p", "h1", "h2", "h3", "h4"])
    
    verse_translations = {}
    footnotes = {}
    
    current_footnote_id = None
    in_footnotes_section = False
    current_speaker = ""
    
    for el in elements:
        text = el.get_text().strip()
        if not text:
            continue
            
        is_fn_marker = False
        fn_classes = el.get("class", [])
        if "nr" in fn_classes or (el.name == "p" and text.startswith("[") and text.split("]")[0][1:].isdigit()):
            is_fn_marker = True
            
        if is_fn_marker:
            in_footnotes_section = True
            digits = re.findall(r"\d+", text)
            if digits:
                current_footnote_id = int(digits[0])
                footnotes[current_footnote_id] = []
            continue
            
        if in_footnotes_section:
            if current_footnote_id is not None:
                if "back to top" in text.lower():
                    current_footnote_id = None
                    continue
                footnotes[current_footnote_id].append(text)
            continue
            
        if el.name in ["h1", "h2", "h3", "h4"]:
            if "chapter" not in text.lower() and "purana" not in text.lower():
                current_speaker = text
            continue
            
        verses, rest_text = extract_verse_numbers_and_text(text)
        if verses:
            if current_speaker:
                rest_text = f"{current_speaker}\n{rest_text}"
                current_speaker = ""
            for v in verses:
                verse_translations[v] = rest_text
        else:
            if verse_translations:
                last_v = max(verse_translations.keys())
                verse_translations[last_v] += "\n" + text
                
    mapped_data = {}
    for v, trans in verse_translations.items():
        # Find all [K] references in trans
        refs = [int(r) for r in re.findall(r"\[(\d+)\]", trans)]
        purport_parts = []
        for r in refs:
            if r in footnotes:
                purport_parts.append(f"Footnote [{r}]:\n" + "\n".join(footnotes[r]))
        
        purport_str = "\n\n".join(purport_parts)
        mapped_data[v] = {
            "translation": trans,
            "purport": purport_str
        }
        
    return mapped_data


def verse_json_from_html(path: Path, book: int, chapter: int, verse_num: int, translation_map: dict) -> dict:
    html = path.read_text(encoding="utf-8")
    verse_ref = f"{book}.{chapter}.{verse_num}"
    devanagari = devanagari_from_html(html)

    english_devnagari = ""
    verse_syn: list[str] = []
    try:
        analysis_text = extract_analysis_text(html, verse_ref)
        parsed = parse_data(analysis_text)
        if parsed:
            lines = parsed[0]["lines"]
            english_devnagari = "".join(line["line_text"].strip() for line in lines)
            for line in lines:
                for token in line["tokens"]:
                    verse_syn.append(token["surface"])
    except Exception:
        pass

    translation = ""
    purport = ""
    if verse_num in translation_map:
        translation = translation_map[verse_num].get("translation", "")
        purport = translation_map[verse_num].get("purport", "")

    return {
        "verse": str(verse_num),
        "devanagari": devanagari,
        "english_devnagari": english_devnagari,
        "verse_Syn": verse_syn,
        "translation": translation,
        "purport": purport,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html-dir", type=Path, default=Path("data/scraped"))
    parser.add_argument("--out-dir", type=Path, default=Path("json"))
    parser.add_argument("--start-book", type=int, default=2)
    parser.add_argument("--end-book", type=int, default=12)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[int, int], list[tuple[int, Path]]] = {}
    for path in args.html_dir.glob("*_*_*.html"):
        match = HTML_RE.match(path.name)
        if not match:
            continue
        book, chapter, verse_num = (int(part) for part in match.groups())
        if args.start_book <= book <= args.end_book:
            grouped.setdefault((book, chapter), []).append((verse_num, path))

    total_files = 0
    total_verses = 0
    for (book, chapter), verse_paths in sorted(grouped.items()):
        # Try to read and parse cached English translation page
        english_path = args.html_dir / "english_chapters" / f"bhagavata_book{book}_ch{chapter}.html"
        translation_map = {}
        if english_path.exists():
            try:
                translation_map = parse_english_translation(english_path.read_text(encoding="utf-8", errors="replace"))
                # Check for parts: part2, part3, etc.
                part = 2
                while True:
                    part_path = args.html_dir / "english_chapters" / f"bhagavata_book{book}_ch{chapter}_part{part}.html"
                    if part_path.exists():
                        part_map = parse_english_translation(part_path.read_text(encoding="utf-8", errors="replace"))
                        translation_map.update(part_map)
                        part += 1
                    else:
                        break
            except Exception as e:
                print(f"Warning: failed to parse English translation for Book {book} Ch {chapter}: {e}")
        
        rows = [
            verse_json_from_html(path, book, chapter, verse_num, translation_map)
            for verse_num, path in sorted(verse_paths)
        ]
        out_path = args.out_dir / f"bhagavata_book{book}_ch{chapter}.json"
        out_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
        )
        total_files += 1
        total_verses += len(rows)
        print(f"Wrote {out_path} ({len(rows)} verses)")

    print(f"JSON files: {total_files}")
    print(f"Verses: {total_verses}")


if __name__ == "__main__":
    main()
