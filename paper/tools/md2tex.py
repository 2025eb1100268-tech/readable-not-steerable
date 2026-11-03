"""Convert the paper draft (results/paper/DRAFT_v*.md) to paper/main.tex for the TMLR template.

The conversion is mechanical: the text is the draft's, word for word. The only departures, all printed by
the script and listed in paper/BUILD_REPORT.md, are:
- the byline, the [CHECK] note on references, the reference list and the author notes are dropped (the
  references come from refs.bib);
- the abstract's three paragraphs are joined into one (the TMLR template asks for one paragraph);
- figure placeholders become figures; file names in captions become figure references, and the two
  appendix figures get short captions;
- the release placeholder in Section 8.4 is filled per build (anonymized or preprint);
- the [CHECK] marker in Section 8.5 is replaced by the sentence it names only if every reference passed
  the reference check (paper/refs_check.tsv); otherwise the sentence is left out.

Formulas and symbols are typeset from an explicit table below; every entry must match the draft, or the
script stops.

    python paper/tools/md2tex.py results/paper/DRAFT_v3.md paper/main.tex
"""
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NOTES = []

TITLE_BREAK = ": "

# ---- citations: "Author(s), Year" as written in the draft -> bib key ----
CITES = {
    "LeCun, 2022": "lecun2022path",
    "Assran et al., 2023": "assran2023ijepa",
    "Bardes et al., 2024": "bardes2024vjepa",
    "Bardes et al., 2022": "bardes2022vicreg",
    "Piloto et al., 2022": "piloto2022intuitive",
    "Liu & Tegmark, 2021": "liu2021conservation",
    "Mialon et al., 2023": "mialon2023lie",
    "Sanchez-Gonzalez et al., 2020": "sanchezgonzalez2020gns",
    "Li et al., 2021": "li2021fno",
    "Greydanus et al., 2019": "greydanus2019hnn",
    "Alet et al., 2021": "alet2021noether",
    "Locatello et al., 2019": "locatello2019challenging",
    "Higgins et al., 2017": "higgins2017betavae",
    "Perez et al., 2018": "perez2018film",
    "Odena et al., 2016": "odena2016deconv",
    "Shi et al., 2016": "shi2016subpixel",
    "Aitken et al., 2017": "aitken2017checkerboard",
    "Belinkov, 2022": "belinkov2022probing",
    "Hewitt & Liang, 2019": "hewitt2019control",
    "Elazar et al., 2021": "elazar2021amnesic",
    "Bertinetto et al., 2021": "bertinetto2021prereg",
    "Albanie et al., 2022": "albanie2022prereg",
    "Okubo, 1970": "okubo1970",
    "Weiss, 1991": "weiss1991",
    "Ledoit & Wolf, 2004": "ledoit2004",
    "Wiskott & Sejnowski, 2002": "wiskott2002sfa",
    "Kraichnan, 1967": "kraichnan1967",
}

# ---- formulas: exact draft text -> LaTeX. Each must occur at least once. ----
FORMULAS = [
    (r"\[0, 2π\]²", r"$[0, 2\pi]^2$"),
    (r"E(k) ∝ k⁴ exp(−2(k/k₀)²), k₀ ∈ \[2, 5\]", r"$E(k) \propto k^4 \exp(-2(k/k_0)^2)$, $k_0 \in [2, 5]$"),
    ("dE/dt = −2νZ", r"$dE/dt = -2\nu Z$"),
    ("log energy E, log enstrophy Z, log palinstrophy P, log maximum |ω|",
     r"log energy $E$, log enstrophy $Z$, log palinstrophy $P$, log maximum $|\omega|$"),
    ("log dissipation 2νZ", r"log dissipation $2\nu Z$"),
    ("parameter Q (", r"parameter $Q$ ("),
    (r"For a direction d with training-set spread σ\_d, we decode z + α σ\_d d for α",
     r"For a direction $d$ with training-set spread $\sigma_d$, we decode $z + \alpha\sigma_d d$ for $\alpha$"),
    ("amplitude a = log E (loudness) and shape s = log(Z/E)",
     r"amplitude $a = \log E$ (loudness) and shape $s = \log(Z/E)$"),
    ("changes a and leaves s fixed", r"changes $a$ and leaves $s$ fixed"),
    (r"σ\_shape = |Δs| / (|Δa| + |Δs|)",
     r"$\sigma_{\mathrm{shape}} = |\Delta s| / (|\Delta a| + |\Delta s|)$"),
    ("the slopes of a and s against α", r"the slopes of $a$ and $s$ against $\alpha$"),
    ("|median Δa| + |median Δs|", r"$|\mathrm{median}\,\Delta a| + |\mathrm{median}\,\Delta s|$"),
    ("(a or s, whichever", r"($a$ or $s$, whichever"),
    ("its projection on d has", r"its projection on $d$ has"),
    (r"commanded α σ\_d", r"commanded $\alpha\sigma_d$"),
    (r"J = Σ‖y\_α − c · y₀‖² / Σ‖y\_α − y₀‖²",
     r"$J = \sum\|y_\alpha - c\, y_0\|^2 / \sum\|y_\alpha - y_0\|^2$"),
    ("rescaling c of the unsteered output y₀", r"rescaling $c$ of the unsteered output $y_0$"),
    ("τ(ε₁σ₁φ₁ + ε₂σ₂φ₂)/2", r"$\tau(\epsilon_1\sigma_1\phi_1 + \epsilon_2\sigma_2\phi_2)/2$"),
    ("φ₁ is the centred log k", r"$\phi_1$ is the centred $\log k$"),
    ("φ₂ the centred square of φ₁", r"$\phi_2$ the centred square of $\phi_1$"),
    ("σ₁ and σ₂ the training", r"$\sigma_1$ and $\sigma_2$ the training"),
    ("ε₁ and ε₂ standard normal", r"$\epsilon_1$ and $\epsilon_2$ standard normal"),
    ("log Z₀ = log E₀ + log(Z₀/E₀)", r"$\log Z_0 = \log E_0 + \log(Z_0/E_0)$"),
    ("E4 = a/b is", r"E4~$= a/b$ is"),
    ("E4 = 1 means", r"E4~$= 1$ means"),
    ("E4 > 1 means", r"E4~$> 1$ means"),
    ("frame t + k from frames (t − 1, t), k ∈ {1, 4}", r"frame $t + k$ from frames $(t - 1, t)$, $k \in \{1, 4\}$"),
    ("frame t + k;", r"frame $t + k$;"),
    ("(u, v)", r"$(u, v)$"),
    ("estimates g, the average gradient of output loudness a with",
     r"estimates $g$, the average gradient of output loudness $a$ with"),
    ("orthogonal to g, where", r"orthogonal to $g$, where"),
    ("the gradient g is orthogonal", r"the gradient $g$ is orthogonal"),
    ("projecting g out", r"projecting $g$ out"),
    ("cos²(g, PC2)", r"$\cos^2(g, \mathrm{PC2})$"),
    ("projecting out g", r"projecting out $g$"),
    ("g is the mean gradient", r"$g$ is the mean gradient"),
    ("(r = −0.82)", r"($r = -0.82$)"),
    ("Re 286–389", r"$\mathrm{Re}$ 286--389"),
    ("log(Z₀/E₀)", r"$\log(Z_0/E_0)$"),
    ("log Z₀", r"$\log Z_0$"),
    ("log E₀", r"$\log E_0$"),
    ("log P R²", r"$\log P$ $R^2$"),
    ("log ν", r"$\log\nu$"),
    ("spectral shape s", r"spectral shape $s$"),
    ("did not move s", r"did not move $s$"),
    ("does not move s", r"does not move $s$"),
    ("encoded s at", r"encoded $s$ at"),
    ("holds s at", r"holds $s$ at"),
    ("how well s can be read", r"how well $s$ can be read"),
    ("supervised s-direction", r"supervised $s$-direction"),
    ("predicts s almost", r"predicts $s$ almost"),
    ("ψ / u / ω", r"$\psi$ / $u$ / $\omega$"),
    ("ω closer to 1; ψ > 2", r"$\omega$ closer to 1; $\psi > 2$"),
    ("α = ±2", r"$\alpha = \pm 2$"),
    ("±2 standard deviations", r"$\pm 2$ standard deviations"),
    ("±2 SD", r"$\pm 2$ SD"),
    ("±0.5 z-units", r"$\pm 0.5$ $z$-units"),
    ("z-scored", r"$z$-scored"),
]
# applied after FORMULAS, wherever they occur; no match is required
OPTIONAL = [
    ("z-score ", r"$z$-score "),
    ("y₀", r"$y_0$"),
    ("Z₀", r"$Z_0$"),
    ("E₀", r"$E_0$"),
    ("|cos|", r"$|\cos|$"),
    (r"\|cos\|", r"$|\cos|$"),
    ("cos²", r"$\cos^2$"),
    ("ω²", r"$\omega^2$"),
]

# ---- generic patterns, applied after the formula table ----
GREEK = {"α": r"\alpha", "β": r"\beta", "τ": r"\tau", "ν": r"\nu", "ω": r"\omega", "ψ": r"\psi",
         "σ": r"\sigma", "φ": r"\phi", "ε": r"\epsilon", "π": r"\pi", "Δ": r"\Delta", "Σ": r"\Sigma"}
SUPS = {"²": "2", "³": "3", "⁴": "4"}
SYMBOLS = {"×": r"\times", "≥": r"\geq", "≤": r"\leq", "≈": r"\approx", "±": r"\pm", "∝": r"\propto",
           "∈": r"\in", "→": r"\rightarrow", "⊕": r"\oplus", "‖": r"\|", "·": r"\cdot"}


class Inline:
    def __init__(self):
        self.store = []

    def hold(self, latex):
        self.store.append(latex)
        return f"\x00{len(self.store) - 1}\x00"

    def restore(self, s):
        while "\x00" in s:
            s = re.sub(r"\x00(\d+)\x00", lambda m: self.store[int(m.group(1))], s)
        return s


def escape_tt(s):
    return (s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
            .replace("#", r"\#").replace("{", r"\{").replace("}", r"\}"))


FORMULA_HITS = {k: 0 for k, _ in FORMULAS}


def num_math(s):
    return s.replace("−", "-")


def convert_inline(text, ctx):
    il = Inline()
    s = text
    # 1. inline code -> \texttt (file names)
    s = re.sub(r"`([^`]*)`", lambda m: il.hold(r"\texttt{" + escape_tt(m.group(1)) + "}"), s)
    # 2. explicit formulas
    for md, tex in FORMULAS:
        if md in s:
            FORMULA_HITS[md] += s.count(md)
            s = s.replace(md, il.hold(tex))
    for md, tex in OPTIONAL:
        if md in s:
            s = s.replace(md, il.hold(tex))
    # 3. generic math patterns
    s = re.sub(r"σ\\_shape", lambda m: il.hold(r"$\sigma_{\mathrm{shape}}$"), s)
    s = re.sub(r"σ\\_d", lambda m: il.hold(r"$\sigma_d$"), s)
    s = re.sub(r"J\\_T", lambda m: il.hold(r"$J_T$"), s)
    s = re.sub(r"\\?\|r\\?\|", lambda m: il.hold(r"$|r|$"), s)
    s = re.sub(r"\|Δ([as])\|", lambda m: il.hold(rf"$|\Delta {m.group(1)}|$"), s)
    s = re.sub(r"Δ([as])\b", lambda m: il.hold(rf"$\Delta {m.group(1)}$"), s)
    s = s.replace("R²", il.hold(r"$R^2$"))
    s = re.sub(r"(\d+)²", lambda m: il.hold(rf"${m.group(1)}^2$"), s)
    s = re.sub(r"(\d+(?:\.\d+)?) × 10⁻([²³⁴])", lambda m: il.hold(rf"${m.group(1)}\times10^{{-{SUPS[m.group(2)]}}}$"), s)
    s = re.sub(r"10⁻([²³⁴])", lambda m: il.hold(rf"$10^{{-{SUPS[m.group(1)]}}}$"), s)
    s = re.sub(r"(\d+) × (\d+) × (\d+)", lambda m: il.hold(rf"${m.group(1)} \times {m.group(2)} \times {m.group(3)}$"), s)
    s = re.sub(r"(\d+) × (\d+)", lambda m: il.hold(rf"${m.group(1)} \times {m.group(2)}$"), s)
    s = re.sub(r"\b(k|r) = ([−+]?\d[\d.]*)", lambda m: il.hold(f"${m.group(1)} = {num_math(m.group(2))}$"), s)
    s = re.sub(r"([ατν]) = ([−+]?\d[\d.]*)",
               lambda m: il.hold(f"${GREEK[m.group(1)]} = {num_math(m.group(2))}$"), s)
    s = re.sub(r"(?<![\w.])([−+])(\d[\d.]*)", lambda m: il.hold(f"${num_math(m.group(1))}{m.group(2)}$"), s)
    for g, cmd in GREEK.items():
        s = s.replace(g, il.hold(f"${cmd}$"))
    # 4. markdown escapes
    s = s.replace("&#91;", "[").replace(r"\[", "[").replace(r"\]", "]").replace(r"\|", "|").replace(r"\_", "_")
    # 5. citations
    def cite(m):
        key = CITES[m.group(0)]
        return il.hold(r"\citealp{" + key + "}")
    pat = "|".join(re.escape(k) for k in sorted(CITES, key=len, reverse=True))
    s = re.sub(pat, cite, s)
    # 6. cross-references
    s = re.sub(r"\bSections (\d+(?:\.\d+)?)–(\d+(?:\.\d+)?)",
               lambda m: il.hold(rf"Sections~\ref{{sec:{m.group(1)}}}--\ref{{sec:{m.group(2)}}}"), s)
    s = re.sub(r"\bSection (\d+(?:\.\d+)?)\b", lambda m: il.hold(rf"Section~\ref{{sec:{m.group(1)}}}"), s)
    s = re.sub(r"\bAppendix ([A-F])\b", lambda m: il.hold(rf"Appendix~\ref{{app:{m.group(1)}}}"), s)
    s = re.sub(r"\bTable ([1-6]|[A-F]\d)\b(?!\.)", lambda m: il.hold(rf"Table~\ref{{tab:{m.group(1)}}}"), s)
    # 7. escape LaTeX specials in the remaining text
    s = (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
         .replace("_", r"\_").replace("{", r"\{").replace("}", r"\}").replace("~", r"\textasciitilde{}"))
    # 8. quotes, dashes and remaining symbols
    s = re.sub(r'(^|[\s(\[])"', r"\1``", s)
    s = s.replace('"', "''")
    s = s.replace("—", "---").replace("–", "--")
    s = s.replace("−", il.hold("$-$")).replace("′", il.hold("$'$")).replace("§", r"\S{}")
    s = s.replace("✓", r"\ding{51}").replace("✗", r"\ding{55}")
    for u, cmd in SYMBOLS.items():
        s = s.replace(u, il.hold(f"${cmd}$"))
    s = s.replace("á", r"{\'a}").replace("í", r"{\'\i}").replace("ä", r'{\"a}').replace("ö", r'{\"o}')
    # 9. emphasis
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)
    s = il.restore(s)
    s = s.replace("$ $", " ")  # nothing glued
    left = [c for c in s if ord(c) > 127]
    if left:
        raise SystemExit(f"unconverted characters {sorted(set(left))} in {ctx}: {text[:80]}")
    return s


# ---- tables ----
TABLE_SPECS = {  # first header cell -> (column spec, font size)
    "Property": ("l X l", r"\small"),
    "Model": ("l X l", r"\small"),
    "Criterion": ("L{4.3cm} c c L{2.3cm} X", r"\small"),
    "Quantity": ("X c c c", r"\small"),
    "Hypothesis": ("X X X l", r"\small"),
    "τ": ("c c c c c c", r"\small"),
    "Seed|Loudness dial": ("C{0.85cm} C{1.25cm} C{1.0cm} C{0.9cm} C{2.0cm} C{2.0cm} X", r"\footnotesize"),
    "Line of work": ("l r", r"\small"),
    "Quantity (in-distribution test, raw pooled latent)": ("X c c c c c c", r"\footnotesize"),
    "Item": ("X X", r"\small"),
    "Stage": ("c X X", r"\small"),
    "Seed|Shape dial: Δs per unit α, exploration / matched": ("c X c X c c", r"\footnotesize"),
    "Seed|cos²(g, PC2)": ("c c X X X", r"\footnotesize"),
    "When": ("l X", r"\small"),
}


def split_row(line):
    cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
    return [c.strip() for c in cells]


def table_tex(rows, caption, label, ctx):
    header = rows[0]
    key = header[0] if header[0] != "Seed" else f"Seed|{header[1]}"
    spec, size = TABLE_SPECS[key]
    out = []
    if caption is not None:
        out += [r"\begin{table}[htbp]", r"\centering", size, r"\caption{" + caption + "}", r"\label{" + label + "}"]
    else:
        out += [r"\begin{table}[htbp]", r"\centering", size]
    cols = spec.split()
    xs = "".join(r">{\raggedright\arraybackslash}X" if c == "X" else c for c in cols)
    out.append(r"\begin{tabularx}{\linewidth}{" + xs + "}")
    out.append(r"\toprule")
    out.append(" & ".join(convert_inline(c, ctx) for c in header) + r" \\")
    out.append(r"\midrule")
    for r in rows[2:]:
        if len(r) != len(header):
            raise SystemExit(f"row width mismatch in {ctx}: {r}")
        out.append(" & ".join(convert_inline(c, ctx) for c in r) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabularx}")
    out.append(r"\end{table}")
    return "\n".join(out)


# ---- figures ----
APPENDIX_FIGS = {
    "fig2b_e4.pdf": ("B", "fig:B1", r"width=0.62\linewidth",
                     r"E4~$= a/b$, the energy--enstrophy mix of the enstrophy-tracking direction (Section~\ref{sec:5.2}), "
                     r"for the plain and VICReg autoencoders of Figure~\ref{fig:2} on the exploration set: seeds and seed "
                     r"mean, log axis. E4~$= 1$ means the direction tracks $\log Z_0$ alone; the dashed line is the "
                     r"untrained control (E)."),
    "fig7b_v3_dials_all.pdf": ("D", "fig:D1", r"height=0.86\textheight",
                               r"As Figure~\ref{fig:7}, for all 11 scored models: main seeds 100--104, $\tau = 0$ seeds "
                               r"200--202 and untrained seeds 300--302, decoded on the same exploration frame; "
                               r"\ding{51}/\ding{55} give each direction's verdict on both sealed sets (C)."),
}


def figure_tex(line, ctx):
    m = re.match(r">\s*\*\*Figure (\d+)\*\*\s*—\s*`figures/([^`]+)`\.?\s*(.*)", line)
    num, fname, rest = m.group(1), m.group(2), m.group(3)
    extra = ""
    # a parenthetical naming the appendix panel, right after the file name (Figure 2)
    pm = re.match(r"\(the E4 panel is `figures/(fig2b_e4\.pdf)`, Appendix B\)\.\s*(.*)", rest)
    if pm:
        rest = pm.group(2)
        extra = r" The E4 panel is Figure~\ref{fig:B1} (Appendix~\ref{app:B})."
        NOTES.append("Figure 2 caption: the pointer to the E4 panel moved to the end of the caption, as a figure reference.")
    rest = re.sub(r"All 11 models: `figures/fig7b_v3_dials_all\.pdf` \(Appendix D\)\.",
                  "All 11 models: @@FIGD1@@.", rest)
    cap = convert_inline(rest, ctx).replace("@@FIGD1@@", r"Figure~\ref{fig:D1} (Appendix~\ref{app:D})")
    if "@@" in cap or "figures/" in cap:
        raise SystemExit(f"unhandled file reference in caption: {line[:80]}")
    width = r"width=\linewidth"
    return "\n".join([r"\begin{figure}[htbp]", r"\centering", rf"\includegraphics[{width}]{{figures/{fname}}}",
                      r"\caption{" + cap + extra + "}", rf"\label{{fig:{num}}}", r"\end{figure}"])


def appendix_figure(fname):
    _, label, size, cap = APPENDIX_FIGS[fname]
    NOTES.append(f"Appendix figure {label}: caption written for the build (the draft names the file only).")
    return "\n".join([r"\begin{figure}[htbp]", r"\centering", rf"\includegraphics[{size}]{{figures/{fname}}}",
                      r"\caption{" + cap + "}", rf"\label{{{label}}}", r"\end{figure}"])


# ---- references check ----
def references_verified():
    path = REPO / "paper/refs_check.tsv"
    if not path.exists():
        return False, ["refs_check.tsv missing"]
    bad = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["status"] != "verified":
                bad.append(row["key"])
    return not bad, bad


CHECK_REFS = '\\[CHECK: after the reference check, add "and every reference was checked against its source."\\]'
RELEASE = r"\[Anonymized repository link for review.\]"


def convert(md):
    lines = md.splitlines()
    out = []
    title = lines[0].lstrip("# ").strip()
    i = 1
    abstract = []
    in_appendix = False
    pending_app_fig = {"B": "fig2b_e4.pdf", "D": "fig7b_v3_dials_all.pdf"}
    ok_refs, bad_refs = references_verified()

    def para(buf):
        if buf:
            out.append(convert_inline(" ".join(buf), "paragraph"))
            out.append("")

    buf = []
    while i < len(lines):
        line = lines[i]
        if line.startswith("Working draft"):
            NOTES.append("Byline dropped.")
            i += 1
            continue
        if line.startswith("## Abstract"):
            i += 1
            while not lines[i].startswith("## "):
                if lines[i].strip():
                    abstract.append(lines[i].strip())
                i += 1
            NOTES.append(f"Abstract: {len(abstract)} paragraphs joined into one (TMLR template).")
            continue
        if line.startswith("## References"):
            para(buf); buf = []
            while not lines[i].startswith("## Appendices"):
                i += 1
            NOTES.append("Reference list and its [CHECK] note dropped; references come from refs.bib.")
            out.append(r"\label{endofmaintext}")
            out.append(r"\bibliography{refs}")
            out.append(r"\bibliographystyle{tmlr}")
            out.append("")
            out.append(r"\clearpage")
            out.append(r"\appendix")
            out.append(r"\phantomsection\label{startofappendix}")
            out.append(r"\counterwithin*{table}{section}\renewcommand{\thetable}{\thesection\arabic{table}}"
                       r"\renewcommand{\theHtable}{\thesection.\arabic{table}}")
            out.append(r"\counterwithin*{figure}{section}\renewcommand{\thefigure}{\thesection\arabic{figure}}"
                       r"\renewcommand{\theHfigure}{\thesection.\arabic{figure}}")
            in_appendix = True
            i += 1
            continue
        if line.startswith("### Author notes") or (line.strip() == "---" and in_appendix):
            NOTES.append("Author notes dropped.")
            break
        m = re.match(r"^(#{2,3}) (\d+(?:\.\d+)?|[A-F]) (.*)$", line)
        if m:
            para(buf); buf = []
            num, head = m.group(2), convert_inline(m.group(3), "heading")
            if re.fullmatch(r"[A-F]", num):
                out.append(rf"\section{{{head}}}\label{{app:{num}}}")
                if num in pending_app_fig:
                    out.append(appendix_figure(pending_app_fig[num]))
            elif "." in num:
                out.append(rf"\subsection{{{head}}}\label{{sec:{num}}}")
            else:
                out.append(rf"\section{{{head}}}\label{{sec:{num}}}")
            out.append("")
            i += 1
            continue
        if line.startswith("|"):
            para(buf); buf = []
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            j = i
            while j < len(lines) and not lines[j].strip():
                j += 1
            caption, label = None, None
            cm = re.match(r"^\*Table ([0-9]+|[A-F][0-9])\. (.*)\*$", lines[j].strip()) if j < len(lines) else None
            if cm:
                caption = convert_inline(cm.group(2), "caption")
                label = f"tab:{cm.group(1)}"
                i = j + 1
            out.append(table_tex(rows, caption, label, f"table {label}"))
            out.append("")
            continue
        if line.startswith("> **Figure"):
            para(buf); buf = []
            out.append(figure_tex(line, "figure"))
            out.append("")
            i += 1
            continue
        if re.match(r"^- ", line) or re.match(r"^\d+\. ", line):
            para(buf); buf = []
            env = "itemize" if line.startswith("- ") else "enumerate"
            out.append(rf"\begin{{{env}}}")
            while i < len(lines) and (re.match(r"^- ", lines[i]) or re.match(r"^\d+\. ", lines[i])):
                item = re.sub(r"^(- |\d+\. )", "", lines[i])
                out.append(r"\item " + convert_inline(item, "list item"))
                i += 1
            out.append(rf"\end{{{env}}}")
            out.append("")
            continue
        if not line.strip():
            para(buf); buf = []
            i += 1
            continue
        if CHECK_REFS in line:
            if ok_refs:
                line = line.replace(" " + CHECK_REFS, "").replace(
                    "which had not written the manuscript.",
                    "which had not written the manuscript, and every reference was checked against its source.")
                NOTES.append("Section 8.5: reference check passed; the sentence was added and the [CHECK] marker removed.")
            else:
                line = line.replace(" " + CHECK_REFS, "")
                NOTES.append("Section 8.5: reference check NOT passed (" + ", ".join(bad_refs) +
                             "); the sentence was left out and the marker removed.")
        if RELEASE in line:
            line = line.replace(RELEASE, "@@RELEASE@@")
        buf.append(line.strip())
        i += 1
    para(buf)

    body = "\n".join(out).replace("@@RELEASE@@", r"\ReleaseStatement")
    abstract_tex = convert_inline(" ".join(abstract), "abstract")
    t = title.split(TITLE_BREAK, 1)
    title_tex = convert_inline(t[0], "title") + r":\\" + convert_inline(t[1], "title")
    missing = [k for k, n in FORMULA_HITS.items() if n == 0]
    if missing:
        raise SystemExit(f"formula table entries not found in the draft: {missing}")
    return title_tex, abstract_tex, body


PREAMBLE = r"""%% Generated by paper/tools/md2tex.py from {src}. Edit the draft or the converter, not this file.
%% Two builds from this one source:
%%   anonymized (TMLR submission): tectonic main.tex                     (or pdflatex main)
%%   named preprint:               pdflatex "\def\PREPRINT{{}}\input{{main}}"   (paper/build.py does both)
\documentclass[10pt]{{article}}
\ifdefined\PREPRINT
  \usepackage[preprint]{{tmlr}}
\else
  \usepackage{{tmlr}}
\fi
\usepackage{{amsmath,amssymb}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{tabularx}}
\usepackage{{array}}
\newcolumntype{{C}}[1]{{>{{\centering\arraybackslash}}p{{#1}}}}
\newcolumntype{{L}}[1]{{>{{\raggedright\arraybackslash}}p{{#1}}}}
\usepackage{{pifont}}
\usepackage[section]{{placeins}}
\usepackage{{url}}
\usepackage[hidelinks]{{hyperref}}

\title{{{title}}}

\ifdefined\PREPRINT
  \author{{\name {author_name} \email {author_email} \\
          \addr {author_addr}}}
  \hypersetup{{pdftitle={{{pdftitle}}}, pdfauthor={{{author_name}}}, pdfsubject={{}}, pdfkeywords={{}}, pdfcreator={{}}, pdfproducer={{}}}}
  \newcommand{{\ReleaseStatement}}{{{release_preprint}}}
\else
  \author{{}}
  \hypersetup{{pdftitle={{{pdftitle}}}, pdfauthor={{}}, pdfsubject={{}}, pdfkeywords={{}}, pdfcreator={{}}, pdfproducer={{}}}}
  \newcommand{{\ReleaseStatement}}{{{release_anon}}}
\fi

\begin{{document}}

\maketitle

\begin{{abstract}}
{abstract}
\end{{abstract}}

{body}
\end{{document}}
"""

AUTHOR = {  # the named preprint only; confirm before posting
    "author_name": "Manvendra Modgil",
    "author_email": "manvendramodgil.ai@gmail.com",
    "author_addr": "Independent researcher",
}
RELEASE_ANON = "An anonymized copy of the repository is provided as supplementary material."
RELEASE_PREPRINT = r"\textbf{[Repository link to be added before posting.]}"


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    title, abstract, body = convert(src.read_text(encoding="utf-8"))
    pdftitle = re.sub(r"\\\\", " ", title).replace(":", ":")
    tex = PREAMBLE.format(src=src.as_posix().replace(str(REPO.as_posix()) + "/", ""), title=title,
                          pdftitle=pdftitle.replace(r"\\", " "), abstract=abstract, body=body,
                          release_anon=RELEASE_ANON, release_preprint=RELEASE_PREPRINT, **AUTHOR)
    dst.write_text(tex, encoding="utf-8", newline="\n")
    for n in dict.fromkeys(NOTES):
        print("note:", n)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
