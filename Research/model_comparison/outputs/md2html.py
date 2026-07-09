#!/usr/bin/env python3
"""Convert report_explained.md to a self-contained HTML file (images inlined as base64)."""
import base64
import mimetypes
import re
from pathlib import Path

import markdown

here = Path(__file__).resolve().parent
src = here / "report_explained.md"
out = here / "report_explained.html"

text = src.read_text()


def inline_image(match):
    alt, path = match.group(1), match.group(2)
    img = (here / path).resolve()
    if not img.exists():
        return match.group(0)
    mime = mimetypes.guess_type(str(img))[0] or "image/png"
    data = base64.b64encode(img.read_bytes()).decode()
    return f'![{alt}](data:{mime};base64,{data})'


# inline ![alt](path) images
text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', inline_image, text)

body = markdown.markdown(
    text,
    extensions=["tables", "fenced_code", "toc", "sane_lists"],
)

css = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  font-size: 11pt; line-height: 1.5; color: #1a1a1a;
  max-width: 820px; margin: 0 auto; padding: 0 8px;
}
h1 { font-size: 22pt; margin: 0 0 .4em; }
h2 { font-size: 16pt; margin: 1.6em 0 .5em; padding-top: .3em;
     border-top: 2px solid #e2e2e2; }
h3 { font-size: 12.5pt; margin: 1.2em 0 .4em; }
h2, h3 { page-break-after: avoid; }
p { margin: .5em 0; }
img { max-width: 100%; height: auto; display: block; margin: 1em auto;
      page-break-inside: avoid; }
table { border-collapse: collapse; width: 100%; margin: 1em 0;
        font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 5px 8px; text-align: left; }
th { background: #f2f2f2; }
tr:nth-child(even) td { background: #fafafa; }
code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px;
       font-size: .9em; }
hr { border: none; border-top: 1px solid #ddd; margin: 2em 0; }
a { color: #1a5fb4; text-decoration: none; }
blockquote { border-left: 3px solid #ccc; margin: 1em 0; padding-left: 1em;
             color: #555; }
"""

html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>report_explained</title>
<style>{css}</style></head><body>{body}</body></html>"""

out.write_text(html)
print("wrote", out)
