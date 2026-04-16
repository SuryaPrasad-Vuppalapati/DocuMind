import os
import json
import re
import pymupdf


def clean_text(text, issues):
    """Clean raw PDF text and track which issues were encountered."""
    text = text.strip()

    # Track mid-sentence line breaks (single \n not adjacent to another \n)
    line_break_matches = re.findall(r"(?<!\n)\n(?!\n)", text)
    if line_break_matches:
        issues["line_breaks"] += len(line_break_matches)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Track excessive whitespace (2+ spaces or tabs)
    whitespace_matches = re.findall(r"[ \t]{2,}", text)
    if whitespace_matches:
        issues["excess_whitespace"] += len(whitespace_matches)
    text = re.sub(r"[ \t]+", " ", text)

    # Track excessive blank lines (3+ consecutive newlines)
    blank_line_matches = re.findall(r"\n{3,}", text)
    if blank_line_matches:
        issues["excess_blank_lines"] += len(blank_line_matches)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def build_issues_section(issues, empty_pages):
    """Build the Issues Encountered section dynamically from runtime data."""
    lines = []

    if issues["line_breaks"] > 0:
        lines.append(
            f"- **Mid-Sentence Line Breaks** ({issues['line_breaks']:,} "
            f"occurrences): Raw PDF text breaks lines mid-sentence wherever "
            f"the PDF renderer wraps text. Fixed by replacing single `\\n` "
            f"with a space while preserving paragraph boundaries (`\\n\\n`)."
        )

    if issues["excess_whitespace"] > 0:
        lines.append(
            f"- **Excessive Whitespace & Tabs** "
            f"({issues['excess_whitespace']:,} occurrences): Redundant spaces "
            f"and tab characters introduced by the PDF layout engine. "
            f"Collapsed all runs of spaces/tabs to a single space."
        )

    if issues["excess_blank_lines"] > 0:
        lines.append(
            f"- **Excessive Blank Lines** "
            f"({issues['excess_blank_lines']:,} occurrences): Pages produced "
            f"3+ consecutive newlines, fragmenting paragraphs. "
            f"Collapsed to double newlines."
        )

    if empty_pages:
        lines.append(
            f"- **Empty / Near-Empty Pages** ({len(empty_pages)} pages): "
            f"Pages with fewer than 50 characters — likely full-page figures, "
            f"part-title pages, or scanned image-only pages with no text "
            f"layer. Flagged below. Future work: apply OCR (e.g. Tesseract)."
        )

    if not lines:
        lines.append("- No issues detected during extraction.")

    return "\n".join(lines)


def extract_from_pdfs(pdf_dir, output_json, sample_json, report_md):
    corpus = []

    if not os.path.exists(pdf_dir):
        print(f"Directory {pdf_dir} does not exist.")
        return

    num_docs = 0
    total_pages = 0
    total_chars = 0
    empty_pages = []
    issues = {
        "line_breaks": 0,
        "excess_whitespace": 0,
        "excess_blank_lines": 0,
    }

    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(pdf_dir, filename)
            print(f"Processing: {filename}")
            num_docs += 1

            try:
                doc = pymupdf.open(filepath)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    raw_text = page.get_text()
                    text = clean_text(raw_text, issues)

                    char_count = len(text)
                    total_pages += 1
                    total_chars += char_count

                    if char_count < 50:
                        empty_pages.append(
                            f"{filename} - Page {page_num + 1} "
                            f"(Chars: {char_count})"
                        )

                    if text:
                        entry = {
                            "source": filename,
                            "page": page_num + 1,
                            "char_count": char_count,
                            "text": text,
                        }
                        corpus.append(entry)
                doc.close()
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    out_dir = os.path.dirname(output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    sample_corpus = corpus[:10] if len(corpus) > 10 else corpus
    with open(sample_json, "w", encoding="utf-8") as f:
        json.dump(sample_corpus, f, indent=2, ensure_ascii=False)

    avg_chars = total_chars / total_pages if total_pages > 0 else 0

    stats_msg = f"""
##################################
# CORPUS STATS
##################################
Number of documents (PDFs)        : {num_docs}
Total pages                       : {total_pages}
Total characters                  : {total_chars}
Average characters per page       : {avg_chars:.2f}
Empty or near-empty pages (< 50)  : {len(empty_pages)}
##################################
"""
    print(stats_msg)

    issues_section = build_issues_section(issues, empty_pages)

    empty_pages_section = ""
    if empty_pages:
        for p in empty_pages:
            empty_pages_section += f"- {p}\n"
    else:
        empty_pages_section = "None\n"

    report_content = f"""# Extraction Report

## Issues Encountered & Handled
{issues_section}

## Corpus Stats
- **Number of documents (PDFs)**: {num_docs}
- **Total pages**: {total_pages}
- **Total characters**: {total_chars:,}
- **Average characters per page**: {avg_chars:.2f}
- **Empty or near-empty pages (< 50 chars)**: {len(empty_pages)}

## Empty or Near-Empty Pages Flagged
{empty_pages_section}"""

    with open(report_md, "w", encoding="utf-8") as f:
        f.write(report_content)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    pdf_dir = os.path.join(base_dir, "data", "pdfs")
    output_json = os.path.join(base_dir, "data", "corpus.json")
    sample_json = os.path.join(base_dir, "data", "sample.json")
    report_md = os.path.join(script_dir, "EXTRACTION_REPORT.md")

    extract_from_pdfs(pdf_dir, output_json, sample_json, report_md)
