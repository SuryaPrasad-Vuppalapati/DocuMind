# Extraction Report

## Issues Encountered & Handled
- **Mid-Sentence Line Breaks** (82,911 occurrences): Raw PDF text breaks lines mid-sentence wherever the PDF renderer wraps text. Fixed by replacing single `\n` with a space while preserving paragraph boundaries (`\n\n`).
- **Excessive Whitespace & Tabs** (753 occurrences): Redundant spaces and tab characters introduced by the PDF layout engine. Collapsed all runs of spaces/tabs to a single space.
- **Empty / Near-Empty Pages** (24 pages): Pages with fewer than 50 characters — likely full-page figures, part-title pages, or scanned image-only pages with no text layer. Flagged below. Future work: apply OCR (e.g. Tesseract).

## Corpus Stats
- **Number of documents (PDFs)**: 2
- **Total pages**: 1099
- **Total characters**: 2,388,009
- **Average characters per page**: 2172.89
- **Empty or near-empty pages (< 50 chars)**: 24

## Empty or Near-Empty Pages Flagged
- pdf.pdf - Page 2 (Chars: 45)
- pdf.pdf - Page 3 (Chars: 0)
- pdf.pdf - Page 7 (Chars: 0)
- pdf.pdf - Page 27 (Chars: 0)
- pdf.pdf - Page 31 (Chars: 0)
- pdf.pdf - Page 57 (Chars: 0)
- pdf.pdf - Page 127 (Chars: 0)
- pdf.pdf - Page 179 (Chars: 0)
- pdf.pdf - Page 221 (Chars: 0)
- pdf.pdf - Page 275 (Chars: 0)
- pdf.pdf - Page 311 (Chars: 0)
- pdf.pdf - Page 337 (Chars: 0)
- pdf.pdf - Page 367 (Chars: 0)
- pdf.pdf - Page 451 (Chars: 0)
- pdf.pdf - Page 509 (Chars: 0)
- pdf.pdf - Page 545 (Chars: 0)
- pdf.pdf - Page 619 (Chars: 0)
- pdf.pdf - Page 737 (Chars: 0)
- pdf.pdf - Page 761 (Chars: 0)
- pdf.pdf - Page 845 (Chars: 0)
- pdf.pdf - Page 867 (Chars: 0)
- pdf.pdf - Page 1025 (Chars: 0)
- pdf.pdf - Page 1039 (Chars: 0)
- pdf.pdf - Page 1077 (Chars: 0)
