#!/usr/bin/env python3
"""Extract LaTeX prose paragraphs into review-grade-polish segments.json.

Only prose that is safe to language-edit is extracted. Excluded (``skip``):
math environments, tables, tabular, verbatim/lstlisting, pgfplots, thebibliography,
equations, align, and math display. Included: abstract, section body prose,
theorem/lemma/proof/remark/definition statement prose, \item text.
Each segment records src_start/src_end (1-based line numbers) for exact back-fill.
"""
import json, re, sys
from pathlib import Path

TEX = Path(sys.argv[1] if len(sys.argv) > 1 else "theory/main_paper.tex")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "outputs/main_paper_润色/work/segments.json")

lines = TEX.read_text(encoding="utf-8").split("\n")

# Environments whose entire body is NOT editable prose.
SKIP_ENV = {
    "equation", "equation*", "align", "align*", "alignat", "alignat*",
    "gather", "gather*", "multline", "multline*", "eqnarray", "eqnarray*",
    "displaymath", "math", "tabular", "tabular*", "tabularx", "longtable",
    "verbatim", "verbatim*", "lstlisting", "minted", "tikzpicture",
    "pgfplotspicture", "thebibliography",
}
# Environments whose body IS editable prose (statements/proofs).
PROSE_ENV = {"abstract", "theorem", "lemma", "proof", "remark", "corollary",
             "proposition", "definition", "claim", "assumption", "example"}

CITE = re.compile(r"\\cite[tbp]?\{[^}]*\}|\\eqref\{[^}]*\}|\\ref\{[^}]*\}|\\label\{[^}]*\}|\\autoref\{[^}]*\}")
NUM  = re.compile(r"-?\d+(?:\.\d+)?%?")

def strip_inline(s: str) -> str:
    s = re.sub(r"%.*$", "", s)  # comments
    return s.rstrip()

segments = []
cur = []          # list of (lineno, text)
cur_start = None

def flush():
    global cur, cur_start
    if cur:
        txt = " ".join(t.strip() for _, t in cur if t.strip())
        if txt:
            segments.append({
                "text": txt,
                "src_start": cur_start,
                "src_end": cur[-1][0],
            })
    cur = []
    cur_start = None

skip_depth = 0
in_document = False
for i, raw in enumerate(lines, 1):
    s = strip_inline(raw)
    t = s.strip()
    if not in_document:
        if t.startswith(r"\begin{document}"):
            in_document = True
        continue
    if t.startswith(r"\end{document}"):
        flush(); break

    # environment tracking
    m_begin = re.match(r"\\begin\{([^}]+)\}", t)
    m_end = re.match(r"\\end\{([^}]+)\}", t)
    if m_begin and m_begin.group(1) in SKIP_ENV:
        flush(); skip_depth += 1; continue
    if m_end and skip_depth > 0:
        skip_depth -= 1; continue
    if skip_depth > 0:
        continue

    # structural breaks that are not editable prose
    if (t.startswith("\\section") or t.startswith("\\subsection")
        or t.startswith("\\subsubsection") or t.startswith("\\paragraph")
        or t.startswith("\\begin{") or t.startswith("\\end{")
        or t.startswith("\\input") or t.startswith("\\caption")
        or t.startswith("\\includegraphics") or t.startswith("\\centering")
        or t.startswith("\\setlength") or t.startswith("\\small") or t.startswith("\\footnotesize")
        or t.startswith("\\label") or t.startswith("\\bottomrule") or t.startswith("\\toprule")
        or t.startswith("\\midrule") or t.startswith("\\date") or t.startswith("\\maketitle")
        or t == ""):
        flush()
        continue

    # pure math line
    if re.match(r"^\\\[|^\\\]|^\$\$", t):
        flush(); continue

    if cur_start is None:
        cur_start = i
    cur.append((i, s))

flush()

# Build segments.json in the shape lint.py expects.
out_segs = []
for idx, seg in enumerate(segments, 1):
    txt = seg["text"]
    # citations: convert \cite{...} to bracket placeholders so lint's cite checker can see them
    cites = CITE.findall(txt)
    nums = NUM.findall(txt)
    out_segs.append({
        "id": idx,
        "para_index": idx - 1,
        "chapter": "",
        "char_count": len(txt),
        "keywords": [],
        "style": "Normal",
        "is_heading": False,
        "text": txt,
        "revised": "",
        "status": "pending",
        "strategies": [],
        "notes": "",
        "inline_format": [],
        "citations": cites,
        "numbers": nums,
        "src_start": seg["src_start"],
        "src_end": seg["src_end"],
    })

data = {"doc_title": "Auditable Privacy Accounting and Dynamic Semantic Budget Allocation",
        "lang": "en", "source": str(TEX), "segments": out_segs}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"source lines: {len(lines)}")
print(f"prose segments extracted: {len(out_segs)}")
print(f"total prose words: {sum(len(re.findall(r'[A-Za-z]+', s['text'])) for s in out_segs)}")
print(f"written: {OUT}")
