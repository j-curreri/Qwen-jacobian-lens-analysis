"""Render paper/index.html to paper/paper.pdf with WeasyPrint (pip install weasyprint).

The web page uses CSS variables and a dark theme; for print we force the light palette, put literal
colours into the inline SVG, and add page rules.

usage: python scripts/build_pdf.py
"""
import os, re
import weasyprint

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
src = open(os.path.join(ROOT, "paper", "index.html"), encoding="utf-8").read()

LIGHT = {"--bar": "#8a3b12", "--bar-ctl": "#9aa0a6", "--bar-pos": "#2f6f8f"}
for k, v in LIGHT.items():
    src = src.replace(f'fill="var({k})"', f'fill="{v}"')
src = src.replace('stroke="currentColor"', 'stroke="#1d1d1f"')

PRINT_CSS = """
<style>
@page { size: Letter; margin: 2cm 2.1cm 2.2cm;
  @bottom-center { content: counter(page); font: 9pt "Bitstream Charter", "Noto Serif", serif; color: #5f6368; } }
:root, :root:not([data-theme="light"]) {
  --bg: #ffffff; --fg: #1d1d1f; --muted: #5f6368; --rule: #cfcac0; --accent: #8a3b12;
  --code-bg: #f2efe8; --table-head: #f3f0e9; }
body { background: #fff; font: 10.5pt/1.5 "Bitstream Charter", "Noto Serif", "Noto Serif CJK SC", serif; }
main { max-width: none; padding: 0; }
h1 { font-size: 19pt; margin-top: 0; }
h2 { font-size: 13pt; break-after: avoid; }
h3 { font-size: 11pt; break-after: avoid; }
code, .mono, .tok, pre { font-family: "DejaVu Sans Mono", "Noto Sans Mono CJK SC", monospace; }
.tok { white-space: normal; }
table { font-size: 8.6pt; }
.table-wrap { overflow: visible; }
table, figure, pre, .abstract { break-inside: avoid; }
tr { break-inside: avoid; }
a { color: #8a3b12; text-decoration: none; }
svg text { fill: #1d1d1f; font-family: "Noto Sans", sans-serif; }
svg .muted { fill: #5f6368; }
.cjk, .tok { font-family: "DejaVu Sans Mono", "Noto Sans Mono CJK SC", monospace; }
</style>
"""
src = src.replace("</head>", PRINT_CSS + "</head>")
out = os.path.join(ROOT, "paper", "paper.pdf")
weasyprint.HTML(string=src, base_url=os.path.join(ROOT, "paper")).write_pdf(out)
print("wrote", out)
