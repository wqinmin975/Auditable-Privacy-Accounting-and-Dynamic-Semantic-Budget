#!/usr/bin/env python3
"""Back-fill revised prose into the .tex by source line range.

- Reads segments.json; for each segment with a non-empty ``revised`` string,
  replaces the source lines [src_start, src_end] with a LaTeX-safe wrapped
  version of the revised text.
- Wrapping never breaks inside $...$, \\command{...}, or \\cite/\\ref/\\eqref.
- Asserts that every number and every \\cite/\\ref/\\eqref token in the original
  also appears in the revised text (zero-drift guard for G2/G11).
- Processes segments in descending line order so earlier offsets stay valid.
"""
import json, re, sys
from pathlib import Path

SEG = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/main_paper_润色/work/segments.json")
if __name__ == '__main__':
    TEX = Path(sys.argv[2] if len(sys.argv) > 2 else "theory/main_paper.tex")
    BACKUP = TEX.with_suffix(".tex.bak")

    TEX = Path(sys.argv[2] if len(sys.argv) > 2 else "theory/main_paper.tex")
    lines = TEX.read_text(encoding="utf-8").split("\n")
    data = json.loads(SEG.read_text(encoding="utf-8"))

    TOKEN = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})*|\$\$?[^$]*\$\$?")
    NUM = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?(?:e[+-]?\d+)?")
    CMD = re.compile(r"\\(?:cite|ref|eqref|autoref|cref|Cref|label)\{[^{}]*\}")

    def tokens(s):
        return sorted(CMD.findall(s)), sorted(NUM.findall(s))

    def wrap_latex(text: str, width: int = 88) -> str:
        """Wrap at spaces, but treat $...$ and \\cmd{...} as unbreakable atoms."""
        # tokenize into atoms: protected spans first, then words/spaces
        protected = []
        def stash(m):
            protected.append(m.group(0))
            return f"\x00{len(protected)-1}\x00"
        tmp = re.sub(r"\$[^$]*\$|\\(?:cite|ref|eqref|autoref|cref|Cref|label)\{[^{}]*\}|\\(?:emph|textbf|textit|texttt|mathrm)\{[^{}]*\}", stash, text)
        words = tmp.split()
        out, cur = [], ""
        for w in words:
            cand = w if not cur else cur + " " + w
            if len(cand) > width and cur:
                out.append(cur); cur = w
            else:
                cur = cand
        if cur:
            out.append(cur)
        joined = "\n".join(out)
        for i, tok in enumerate(protected):
            joined = joined.replace(f"\x00{i}\x00", tok)
        return joined

    repls = []
    errors = []
    for s in data["segments"]:
        revised = s.get("revised", "")
        if not revised.strip():
            continue
        orig = s["text"]
        a, b = sorted(CMD.findall(orig)), sorted(NUM.findall(orig))
        ra, rb = sorted(CMD.findall(revised)), sorted(NUM.findall(revised))
        if a != ra:
            errors.append(f"seg{s['id']}: command drift\n  orig={a}\n  rev ={ra}")
        if b != rb:
            errors.append(f"seg{s['id']}: number drift\n  orig={b}\n  rev ={rb}")
        repls.append((s["src_start"], s["src_end"], wrap_latex(revised)))

    if errors:
        print("ZERO-DRIFT GUARD FAILED — no file written:")
        print("\n".join(errors))
        sys.exit(1)

    # backup once
    if not BACKUP.exists():
        BACKUP.write_text("\n".join(lines), encoding="utf-8")

    for start, end, newtext in sorted(repls, reverse=True):
        lines[start-1:end] = newtext.split("\n")

    TEX.write_text("\n".join(lines), encoding="utf-8")
    print(f"applied {len(repls)} revised segments to {TEX}")
    print(f"backup at {BACKUP}")
