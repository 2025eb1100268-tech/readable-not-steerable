"""Build the paper: both PDFs, the arXiv source tarball, the anonymized supplementary zip, and the checks.
Any failed check stops it.

    python paper/build.py [results/paper/DRAFT_v4.md]

Needs TeX Live (pdflatex and bibtex on PATH, or TEXLIVE_BIN set to their folder) and the `pypdf` package.

Steps:
1. references: every entry of refs.bib is `verified` in refs_check.tsv and unchanged since it was checked;
2. convert the draft to main.tex (tools/md2tex.py), applying the build-time moves to the appendix;
3. pdfLaTeX safety: main.tex, main_preprint.tex and refs.bib are pure ASCII (no raw Unicode);
4. number lint (tools/numlint.py): every numeric literal in main.tex traces to NUMBERS.md or the allow-list;
5. copy the figure PDFs from results/paper/figures/;
6. anonymized build: pdflatex, bibtex, pdflatex, pdflatex;
7. arXiv source: bibtex once for the preprint's .bbl, then out/arxiv_source.tar.gz (main.tex with the preprint
   switch set, main.bbl, style files, figures), unpacked into a clean folder and compiled as arXiv does:
   pdflatex three times, no bibtex. That PDF is the preprint;
8. reset the PDF metadata: title only (anonymized), title and author (preprint); no XMP;
9. build the anonymized supplementary zip (tools/make_supplementary.py);
10. anonymity check of the anonymized PDF and the zip;
11. page count of the main text; write out/build_summary.txt.
"""
import csv
import gzip
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent
OUT = PAPER / "out"
sys.path.insert(0, str(PAPER / "tools"))

import anonymity  # noqa: E402
import check_refs  # noqa: E402
import make_supplementary  # noqa: E402
import md2tex  # noqa: E402
import numlint  # noqa: E402

TITLE = "Readable Is Not Steerable: Pre-Registered Tests of Physical Structure in Learned Fluid Representations"
STYLE = ("tmlr.sty", "fancyhdr.sty")
SUMMARY = []
ENV = dict(os.environ, SOURCE_DATE_EPOCH="1767225600", FORCE_SOURCE_DATE="1")  # fixed date: 2026-01-01


def say(msg):
    print(msg)
    SUMMARY.append(msg)


def fail(msg):
    say("BUILD FAILED: " + msg)
    (OUT / "build_summary.txt").write_text("\n".join(SUMMARY) + "\n", encoding="utf-8")
    sys.exit(1)


def step_refs():
    entries = check_refs.parse_bib((PAPER / "refs.bib").read_text(encoding="utf-8"))
    with open(PAPER / "refs_check.tsv", encoding="utf-8") as fh:
        rows = {r["key"]: r for r in csv.DictReader(fh, delimiter="\t")}
    bad = []
    for key, e in entries.items():
        r = rows.get(key)
        sha = hashlib.sha256(e["raw"].encode("utf-8")).hexdigest()[:16]
        if r is None or r["status"] != "verified" or r["entry_sha256"] != sha:
            bad.append(key)
    say(f"references: {len(entries) - len(bad)} of {len(entries)} verified and unchanged since the check"
        + (f"; not verified: {', '.join(bad)}" if bad else ""))
    return not bad


def step_convert(draft):
    md2tex.NOTES.clear()
    title, abstract, body = md2tex.convert(Path(draft).read_text(encoding="utf-8"))
    src = Path(draft).resolve().relative_to(REPO).as_posix()
    tex = md2tex.PREAMBLE.format(src=src, title=title, pdftitle=TITLE, abstract=abstract, body=body,
                                 release_anon=md2tex.RELEASE_ANON, release_preprint=md2tex.RELEASE_PREPRINT,
                                 **md2tex.AUTHOR)
    (PAPER / "main.tex").write_text(tex, encoding="utf-8", newline="\n")
    for n in dict.fromkeys(md2tex.NOTES):
        say("conversion: " + n)


def step_ascii():
    for f in ("main.tex", "main_preprint.tex", "refs.bib"):
        data = (PAPER / f).read_bytes()
        bad = sum(b > 127 for b in data)
        if bad:
            fail(f"{f} has {bad} non-ASCII bytes; pdfLaTeX needs macros, not raw Unicode")
    say("pdfLaTeX safety: main.tex, main_preprint.tex and refs.bib are pure ASCII")


def step_lint():
    found, missing, problems = numlint.lint((PAPER / "main.tex").read_text(encoding="utf-8"), "tex")
    for p in problems:
        say("numlint: " + p)
    for tok, ctx in missing:
        say(f"numlint: not traced {tok!r}: ...{ctx}...")
    say(f"number lint: {len(found)} numeric literals checked, {len(missing)} not traced")
    if missing or problems:
        fail("number lint")


def used_figures():
    return sorted(set(re.findall(r"figures/([\w.]+\.pdf)", (PAPER / "main.tex").read_text(encoding="utf-8"))))


def step_figures():
    src = REPO / "results/paper/figures"
    (PAPER / "figures").mkdir(exist_ok=True)
    for f in used_figures():
        shutil.copyfile(src / f, PAPER / "figures" / f)
        assert (src / f).read_bytes() == (PAPER / "figures" / f).read_bytes()
    say(f"figures: {len(used_figures())} PDFs copied from results/paper/figures/")


def tool(name):
    d = os.environ.get("TEXLIVE_BIN")
    exe = shutil.which(name, path=d) if d else shutil.which(name)
    if not exe:
        fail(f"{name} not found: install TeX Live or set TEXLIVE_BIN")
    return exe


def tex_version():
    out = subprocess.run([tool("pdflatex"), "--version"], capture_output=True, text=True).stdout
    return out.splitlines()[0]


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=ENV, errors="replace")
    if r.returncode != 0:
        print(r.stdout[-4000:])
        fail(f"{' '.join(Path(c).name if i == 0 else c for i, c in enumerate(cmd))} failed in {cwd.name}")
    return r


def fresh(outdir, files):
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    for f in files:
        shutil.copyfile(PAPER / f, outdir / f)
    (outdir / "figures").mkdir()
    for f in used_figures():
        shutil.copyfile(PAPER / "figures" / f, outdir / "figures" / f)


def check_log(outdir, job):
    log = (outdir / f"{job}.log").read_text(encoding="utf-8", errors="replace")
    undefined = re.findall(r"(?:Reference|Citation) `([^']+)' .*undefined", log)
    if undefined:
        fail(f"{job}: undefined references or citations: {sorted(set(undefined))}")
    if re.search(r"Rerun to get|Label\(s\) may have changed", log):
        fail(f"{job}: cross-references not settled after the last pass")
    return re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log)


def pdflatex(job, cwd):
    return run([tool("pdflatex"), "-interaction=nonstopmode", "-halt-on-error", f"{job}.tex"], cwd)


def settle(job, cwd, max_passes=5):
    """Rerun pdflatex until the log asks for no rerun (as arXiv's AutoTeX does); return the number of passes."""
    for n in range(1, max_passes + 1):
        pdflatex(job, cwd)
        log = (cwd / f"{job}.log").read_text(encoding="utf-8", errors="replace")
        if not re.search(r"Rerun to get|Label\(s\) may have changed", log):
            return n
    return max_passes


PASSES = {}


def build_anonymized():
    d = OUT / "tex_anonymized"
    fresh(d, ("main.tex", "tmlr.bst", "refs.bib") + STYLE)
    pdflatex("main", d)
    run([tool("bibtex"), "main"], d)
    PASSES["anonymized"] = 1 + settle("main", d)
    return d / "main.pdf", check_log(d, "main")


def tar_add(tf, name, data):
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode, info.uid, info.gid, info.uname, info.gname = len(data), 1767225600, 0o644, 0, 0, "", ""
    tf.addfile(info, io.BytesIO(data))


def build_arxiv():
    d = OUT / "tex_preprint_bbl"
    fresh(d, ("main.tex", "main_preprint.tex", "tmlr.bst", "refs.bib") + STYLE)
    pdflatex("main_preprint", d)
    run([tool("bibtex"), "main_preprint"], d)
    members = {
        "main.tex": b"\\def\\PREPRINT{}%% named preprint build (arXiv)\n" + (PAPER / "main.tex").read_bytes(),
        "main.bbl": (d / "main_preprint.bbl").read_bytes(),
    }
    for f in STYLE:
        members[f] = (PAPER / f).read_bytes()
    for f in used_figures():
        members[f"figures/{f}"] = (PAPER / "figures" / f).read_bytes()
    tgz = OUT / "arxiv_source.tar.gz"
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for name in sorted(members):
            tar_add(tf, name, members[name])
    with open(tgz, "wb") as fh, gzip.GzipFile(filename="", fileobj=fh, mode="wb", mtime=0) as gz:
        gz.write(raw.getvalue())
    # compile exactly as arXiv does: unpack into an empty folder, pdflatex until settled, no bibtex
    a = OUT / "tex_arxiv"
    if a.exists():
        shutil.rmtree(a)
    a.mkdir(parents=True)
    with tarfile.open(tgz) as tf:
        tf.extractall(a, filter="data")
    PASSES["arXiv"] = settle("main", a)
    return tgz, sorted(members), a / "main.pdf", check_log(a, "main")


def reset_metadata(pdf, dest, info):
    from pypdf import PdfReader, PdfWriter
    r = PdfReader(str(pdf))
    w = PdfWriter(clone_from=r)
    if "/Metadata" in w._root_object:
        del w._root_object["/Metadata"]
    w._info = None
    w.add_metadata(info)
    with open(dest, "wb") as fh:
        w.write(fh)
    r2 = PdfReader(str(dest))
    return {k: str(v) for k, v in (r2.metadata or {}).items()}


def label_page(auxdir, name, label):
    aux = (auxdir / f"{name}.aux").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\\newlabel\{" + label + r"\}\{\{[^}]*\}\{(\d+)\}", aux)
    return int(m.group(1)) if m else None


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    draft = sys.argv[1] if len(sys.argv) > 1 else "results/paper/DRAFT_v4.md"
    OUT.mkdir(exist_ok=True)
    SUMMARY.clear()
    say(f"draft: {draft}")
    if not step_refs():
        say("references: the sentence in Section 8.5 will be left out")
    step_convert(REPO / draft)
    step_ascii()
    step_lint()
    step_figures()
    say(f"TeX: {tex_version()}")
    anon_pdf, anon_over = build_anonymized()
    tgz, members, arx_pdf, arx_over = build_arxiv()
    say(f"LaTeX: anonymized (pdflatex, bibtex, then pdflatex until settled: {PASSES['anonymized']} passes in all) and "
        f"arXiv-style preprint (pdflatex until settled from the unpacked tarball, no bibtex: {PASSES['arXiv']} passes) "
        f"compiled; overfull boxes: anonymized {len(anon_over)}, preprint {len(arx_over)}"
        + (f" (largest {max(map(float, anon_over + arx_over)):.1f}pt)" if anon_over + arx_over else ""))
    say(f"arXiv source: {tgz.name}, {len(members)} files ({', '.join(m for m in members if not m.startswith('figures/'))} "
        f"+ {sum(m.startswith('figures/') for m in members)} figures), {tgz.stat().st_size / 1024:.0f} KB, sha256 {sha(tgz)}")
    a_info = reset_metadata(anon_pdf, OUT / "anonymized.pdf", {"/Title": TITLE})
    p_info = reset_metadata(arx_pdf, OUT / "preprint.pdf", {"/Title": TITLE, "/Author": md2tex.AUTHOR["author_name"]})
    say(f"PDF metadata: anonymized {a_info}; preprint {p_info}")
    sup = make_supplementary.build(OUT / "supplementary_anonymized.zip")
    say(f"supplementary: {sup['files']} files, {sup['bytes'] / 2**20:.1f} MB; excluded {len(sup['excluded'])}; "
        f"anonymized by substitution: {', '.join(f'{e} ({k})' for e, k in sup['edited'])}")
    words, generic = anonymity.needles()
    say(f"anonymity needles: {len(words)} identity strings found at run time + {', '.join(generic)}")
    a = anonymity.check_pdf(OUT / "anonymized.pdf", words, generic)
    z = anonymity.check_zip(OUT / "supplementary_anonymized.zip", words, generic)
    say(f"anonymity, anonymized PDF: {len(a['hits'])} hits; XMP metadata present: {a['xmp']}")
    say(f"anonymity, supplementary zip ({z['files']} files): {len(z['hits'])} hits")
    for h in a["hits"] + z["hits"]:
        say(f"  hit: {h}")
    p = anonymity.check_pdf(OUT / "preprint.pdf", words, generic)
    say(f"preprint (named, expected to identify): {sorted(set(w for _, w, _ in p['hits']))}")
    for name, auxdir, job, pages in (("anonymized", OUT / "tex_anonymized", "main", a["pages"]),
                                     ("preprint", OUT / "tex_arxiv", "main", p["pages"])):
        end, app = label_page(auxdir, job, "endofmaintext"), label_page(auxdir, job, "startofappendix")
        say(f"length, {name}: main text ends on page {end}; appendices start on page {app}; total {pages} pages")
    for f in ("anonymized.pdf", "preprint.pdf", "supplementary_anonymized.zip"):
        say(f"sha256 {f}: {sha(OUT / f)}")
    (OUT / "build_summary.txt").write_text("\n".join(SUMMARY) + "\n", encoding="utf-8")
    if a["hits"] or z["hits"]:
        fail("anonymity check")
    print("build: OK")


if __name__ == "__main__":
    main()
