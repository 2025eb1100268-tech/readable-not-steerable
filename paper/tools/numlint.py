"""Number lint for the paper build.

Every numeric literal in the paper's text (title, abstract, body, tables, captions, appendices) must
appear in `results/paper/NUMBERS.md`, or in the committed file named for it in `paper/lint_allow.tsv`.
Otherwise the build fails.

What counts as a numeric literal, and how it is matched:
- Tokens are matched as written, digit for digit: 0.80 does not match 0.8. Thousands separators, the
  percent sign and the form of a power of ten (3 × 10⁻³, 3\\times10^{-3}) are normalized away.
- Signs: a negative literal must match a negative token in NUMBERS.md, a positive or unsigned literal a
  positive or unsigned one. A minus between two operands (t − 1) is an operator, not a sign.
- Ranges (0.57–0.65) are checked endpoint by endpoint.
- Not literals: identifiers that contain digits (S1, H4, PC2, E1, float64, 2D, SHA-256), the number after
  Study, Stage, Section, Table, Figure, Appendix, Phase or entry, sub- and superscript indices (k₀, R²,
  64²), and LaTeX structure (commands, environment arguments, lengths, labels, citations, file names).

The check is token-level: it catches a number that appears nowhere in the pack (a typo or a stale value),
not a number quoted in the wrong context. Context was checked claim by claim in AUDIT_v1.md.

    python paper/tools/numlint.py paper/main.tex
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NUMBERS = REPO / "results/paper/NUMBERS.md"
ALLOW = REPO / "paper/lint_allow.tsv"

MINUS = chr(0x2212)
SUPMINUS = chr(0x207B)
SUPDIGITS = "".join(chr(c) for c in (0x2070, 0xB9, 0xB2, 0xB3, 0x2074, 0x2075, 0x2076, 0x2077, 0x2078, 0x2079))
SUBDIGITS = "".join(chr(0x2080 + i) for i in range(10))
SUP = {c: str(i) for i, c in enumerate(SUPDIGITS)}
ID_WORDS = r"(?:Stud(?:y|ies)|Stages?|Sections?|Tables?|Figures?|Appendi(?:x|ces)|Phases?|entr(?:y|ies)|step)"
SCI = re.compile(r"(?:(?P<m>\d+(?:\.\d+)?)\s*(?:\u00d7|\\times)\s*)?(?<![*\d])10\s*(?:" + SUPMINUS + r"|\^\{?-)"
                 r"(?P<e>[" + SUPDIGITS + r"\d]+)\}?")
NUM = re.compile(r"(?P<sign>[-+])?(?P<num>\d+(?:\.\d+)?(?:\*10\^-\d+)?)")


def _sci(m):
    mant = m.group("m") or "1"
    exp = "".join(SUP.get(c, c) for c in m.group("e"))
    return f" {mant}*10^-{exp} "  # canonical form with no letters, so identifier removal leaves it alone


def sci(t):
    """Powers of ten to one canonical form."""
    return SCI.sub(_sci, t)


def normalize(text):
    """Plain-text normalization shared by markdown and LaTeX sources."""
    t = text.replace(MINUS, "-")
    t = sci(t)
    t = re.sub("[" + SUPDIGITS + SUBDIGITS + "]+", " ", t)  # sub- and superscript indices
    t = re.sub(r"SHA-\d+", " ", t)
    t = re.sub(rf"\b{ID_WORDS}\s+\d+[a-z]?(?:\s*(?:\u2013|--|-|,|and|to)\s*\d+[a-z]?)*", " ", t, flags=re.I)
    t = re.sub(r"\b[A-Za-z][A-Za-z\u2032']*\d+[A-Za-z0-9]*\b", " ", t)  # S1, H4, PC2, E1, float64, A1
    t = re.sub(r"\b\d+[A-Za-z]+\b", " ", t)  # 2D, 3D, 2b
    t = re.sub(r"(?<=\d)\{,\}(?=\d{3})", ",", t)
    t = re.sub(r"(?<![\d.,])(\d{1,3}),(\d{3})(?![\d])", r"\1\2", t)  # thousands separators
    return t


def tokens(text):
    """Return (token, position) pairs after normalization; negative tokens carry a leading '-'."""
    t = normalize(text)
    out = []
    for m in NUM.finditer(t):
        start = m.start("num")
        if m.group("sign") is None and start > 0 and (t[start - 1].isalnum() or t[start - 1] in "._"):
            continue  # part of a word or of a longer number
        s0 = m.start()
        # '-' touching the digits is a sign unless an operand touches it on the left (t-1, x)-2); a spaced
        # minus (t − 1) never reaches here, because the sign must touch the digits
        neg = m.group("sign") == "-" and not (s0 > 0 and (t[s0 - 1].isalnum() or t[s0 - 1] in ")]}"))
        num = m.group("num")
        out.append(("-" + num if neg else num, start))
    return out


def strip_markdown(md):
    md = re.sub(r"`[^`]*`", " ", md)  # inline code: file names
    md = re.sub(r"\]\([^)]*\)", "]", md)  # link targets
    return md


def strip_tex(tex):
    body = tex.split(r"\begin{document}", 1)[-1].split(r"\end{document}", 1)[0]
    body = re.sub(r"(?<!\\)%.*", "", body)  # comments
    body = sci(body)  # before sub/superscripts are dropped below
    for cmd in ("label", "ref", "eqref", "autoref", "cref", "Cref", "pageref", "citep", "citet", "citealp",
                "citealt", "cite", "citeauthor", "citeyear", "includegraphics", "url", "texttt", "href",
                "input", "bibliography", "bibliographystyle", "vspace", "hspace", "setlength", "addtolength",
                "arraystretch", "multicolumn", "cmidrule", "ding", "renewcommand", "setcounter", "rule",
                "resizebox", "phantom", "linespread", "fontsize"):
        body = re.sub(rf"\\{cmd}\*?(?:\[[^\]]*\])*(?:\{{[^{{}}]*(?:\{{[^{{}}]*\}}[^{{}}]*)*\}})+", " ", body)
    body = re.sub(r"\\begin\{[^}]*\}(?:\[[^\]]*\])?(?:\{(?:[^{}]|\{[^{}]*\})*\})*", " ", body)  # env args
    body = re.sub(r"\\end\{[^}]*\}", " ", body)
    body = re.sub(r"-?\d*\.?\d+\s*(?:pt|em|ex|in|cm|mm|bp)\b", " ", body)  # lengths
    body = re.sub(r"\d*\.?\d+\\(?:linewidth|textwidth|columnwidth|hsize)", " ", body)
    body = re.sub(r"[\^_]\{[^{}]*\}", " ", body)  # sub/superscript groups (powers of ten already handled)
    body = re.sub(r"[\^_]\\?[A-Za-z0-9]", " ", body)
    body = body.replace("---", "\u2014").replace("--", "\u2013")
    body = re.sub(r"\\[A-Za-z]+", " ", body)  # remaining command names
    body = body.replace("{", "").replace("}", "").replace("$", "").replace("~", " ").replace("\\", " ")
    return body


def load_allow():
    rows = []
    if ALLOW.exists():
        for line in ALLOW.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            tok, path, note = (line.split("\t") + ["", ""])[:3]
            rows.append((tok.strip(), path.strip(), note.strip()))
    return rows


def pack_tokens(text):
    return {tok for tok, _ in tokens(strip_markdown(text))}


def lint(text, kind):
    pack = pack_tokens(NUMBERS.read_text(encoding="utf-8"))
    allow = {}
    problems = []
    for tok, path, note in load_allow():
        p = REPO / path
        if not p.exists():
            problems.append(f"allow-list: {path} does not exist (token {tok})")
            continue
        if tok not in {t for t, _ in tokens(p.read_text(encoding="utf-8", errors="replace"))}:
            problems.append(f"allow-list: token {tok} not found in {path}")
            continue
        allow[tok] = (path, note)
    body = strip_tex(text) if kind == "tex" else strip_markdown(text)
    found = tokens(body)
    norm = normalize(body)
    missing = []
    for tok, pos in found:
        if tok in pack or tok in allow:
            continue
        ctx = norm[max(0, pos - 50): pos + 30].replace("\n", " ")
        missing.append((tok, ctx))
    return found, missing, problems


def main():
    src = Path(sys.argv[1])
    kind = "tex" if src.suffix == ".tex" else "md"
    found, missing, problems = lint(src.read_text(encoding="utf-8"), kind)
    for p in problems:
        print("ERROR", p)
    for tok, ctx in missing:
        print(f"MISSING {tok!r:>12}  ...{ctx}...")
    print(f"numlint: {len(found)} numeric literals checked, {len(missing)} not traced, "
          f"{len(problems)} allow-list errors")
    sys.exit(1 if missing or problems else 0)


if __name__ == "__main__":
    main()
