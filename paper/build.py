"""Build the paper: both PDFs, the anonymized supplementary zip, and the checks. Any failed check stops it.

    python paper/build.py [results/paper/DRAFT_v3.md]

Needs a TeX engine (Tectonic: the `tectonic` command, or set TECTONIC to its path) and the `pypdf` package.

Steps:
1. references: every entry of refs.bib is `verified` in refs_check.tsv and unchanged since it was checked;
2. convert the draft to main.tex (tools/md2tex.py);
3. number lint (tools/numlint.py): every numeric literal in main.tex traces to NUMBERS.md or the allow-list;
4. copy the figure PDFs from results/paper/figures/;
5. compile the anonymized build (main.tex) and the named preprint (main_preprint.tex);
6. reset the PDF metadata: title only (anonymized), title and author (preprint); no XMP, no dates;
7. build the anonymized supplementary zip (tools/make_supplementary.py);
8. anonymity check of the anonymized PDF and the zip;
9. page count of the main text; write out/build_summary.txt.
"""
import csv
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
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
SUMMARY = []


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


def step_lint():
    found, missing, problems = numlint.lint((PAPER / "main.tex").read_text(encoding="utf-8"), "tex")
    for p in problems:
        say("numlint: " + p)
    for tok, ctx in missing:
        say(f"numlint: not traced {tok!r}: ...{ctx}...")
    say(f"number lint: {len(found)} numeric literals checked, {len(missing)} not traced")
    if missing or problems:
        fail("number lint")


def step_figures():
    src = REPO / "results/paper/figures"
    used = set(re.findall(r"figures/([\w.]+\.pdf)", (PAPER / "main.tex").read_text(encoding="utf-8")))
    (PAPER / "figures").mkdir(exist_ok=True)
    for f in sorted(used):
        shutil.copyfile(src / f, PAPER / "figures" / f)
        assert (src / f).read_bytes() == (PAPER / "figures" / f).read_bytes()
    say(f"figures: {len(used)} PDFs copied from results/paper/figures/")


def tectonic():
    exe = os.environ.get("TECTONIC") or shutil.which("tectonic")
    if not exe:
        fail("no TeX engine: install Tectonic or set TECTONIC")
    return exe


def compile_tex(name, outdir):
    # Tectonic's BibTeX finds the style only next to the input, so each build runs in a self-contained copy
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    for f in ("main.tex", "main_preprint.tex", "tmlr.sty", "tmlr.bst", "fancyhdr.sty", "refs.bib"):
        shutil.copyfile(PAPER / f, outdir / f)
    shutil.copytree(PAPER / "figures", outdir / "figures")
    env = dict(os.environ, SOURCE_DATE_EPOCH="1767225600")  # fixed date: 2026-01-01
    r = subprocess.run([tectonic(), "-X", "compile", f"{name}.tex", "--keep-intermediates", "--keep-logs"],
                       cwd=outdir, capture_output=True, text=True, env=env)
    log = (outdir / f"{name}.log").read_text(encoding="utf-8", errors="replace") if (outdir / f"{name}.log").exists() else ""
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        fail(f"LaTeX failed for {name}.tex")
    undefined = re.findall(r"(?:Reference|Citation) `([^']+)' .*undefined", log)
    overfull = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log)
    if undefined:
        fail(f"{name}: undefined references or citations: {sorted(set(undefined))}")
    return outdir / f"{name}.pdf", overfull


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


def main_text_pages(auxdir, name):
    aux = (auxdir / f"{name}.aux").read_text(encoding="utf-8", errors="replace")
    end = re.search(r"\\newlabel\{endofmaintext\}\{\{[^}]*\}\{(\d+)\}", aux)
    refs = re.search(r"\\newlabel\{startofappendix\}\{\{[^}]*\}\{(\d+)\}", aux)
    return int(end.group(1)), int(refs.group(1)) if refs else None


def main():
    draft = sys.argv[1] if len(sys.argv) > 1 else "results/paper/DRAFT_v3.md"
    OUT.mkdir(exist_ok=True)
    SUMMARY.clear()
    say(f"draft: {draft}")
    if not step_refs():
        say("references: the sentence in Section 8.5 will be left out")
    step_convert(REPO / draft)
    step_lint()
    step_figures()
    anon_pdf, anon_over = compile_tex("main", OUT / "tex_anonymized")
    pre_pdf, pre_over = compile_tex("main_preprint", OUT / "tex_preprint")
    say(f"LaTeX: both builds compiled; overfull boxes: anonymized {len(anon_over)}, preprint {len(pre_over)}"
        + (f" (largest {max(map(float, anon_over)):.1f}pt)" if anon_over else ""))
    a_info = reset_metadata(anon_pdf, OUT / "anonymized.pdf", {"/Title": TITLE})
    p_info = reset_metadata(pre_pdf, OUT / "preprint.pdf", {"/Title": TITLE, "/Author": md2tex.AUTHOR["author_name"]})
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
    end, app = main_text_pages(OUT / "tex_anonymized", "main")
    say(f"length: main text ends on page {end}; references and appendices start after it; "
        f"appendices start on page {app}; total {a['pages']} pages (anonymized build)")
    (OUT / "build_summary.txt").write_text("\n".join(SUMMARY) + "\n", encoding="utf-8")
    if a["hits"] or z["hits"]:
        fail("anonymity check")
    print("build: OK")


if __name__ == "__main__":
    main()
