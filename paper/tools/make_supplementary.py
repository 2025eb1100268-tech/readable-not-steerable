"""Build the anonymized supplementary zip from the committed repository (git HEAD).

Contents: every tracked file at HEAD, except
- paper/ (the LaTeX source names the author in its preprint branch; the PDF is the submission), and
- results/paper/DRAFT_v*.md (working drafts with the author's byline).

Text files are anonymized by substitution; nothing else is changed:
- session scratchpad paths -> <scratchpad>, other Windows home paths -> <home>;
- the author's name, user names and e-mail address -> [anonymized];
- the repository's folder name -> anonymized-project.
Binary files are copied as they are. A README at the root of the zip lists every file that was edited.
Entries carry a fixed timestamp and the zip has no comment, so the zip depends only on HEAD.
"""
import io
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path

from anonymity import REPO, needles

EXCLUDE = (re.compile(r"^paper/"), re.compile(r"^results/paper/DRAFT_v\d+\.md$"))
TEXT_EXT = {".md", ".py", ".yaml", ".yml", ".txt", ".json", ".tsv", ".csv", ".ini", ".cfg", ".toml", ".sh", ""}
FIXED_TIME = (2026, 1, 1, 0, 0, 0)


def substitutions():
    words, _ = needles()
    subs = [
        (re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s`'\"]+[\\/]+AppData[\\/]+Local[\\/]+Temp[\\/]+claude[\\/]+"
                    r"[^\\/\s`'\"]+[\\/]+[0-9a-f-]{36}[\\/]+scratchpad"), "<scratchpad>"),
        (re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s`'\"]+"), "<home>"),
        (re.compile(re.escape(REPO.name), re.I), "anonymized-project"),
    ]
    for w in sorted(words, key=len, reverse=True):
        if w.lower() == REPO.name.lower():
            continue
        subs.append((re.compile(re.escape(w), re.I), "[anonymized]"))
    return subs


def head_files():
    tar_bytes = subprocess.run(["git", "archive", "--format=tar", "HEAD"], capture_output=True, cwd=REPO,
                               check=True).stdout
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO,
                          check=True).stdout.strip()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
        for m in tf.getmembers():
            if m.isfile():
                yield m.name, tf.extractfile(m).read()
    return head


def build(zip_path):
    subs = substitutions()
    edited, excluded, n = [], [], 0
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in head_files():
            if any(p.search(name) for p in EXCLUDE):
                excluded.append(name)
                continue
            if Path(name).suffix.lower() in TEXT_EXT:
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    text = None
                if text is not None:
                    new = text
                    for pat, rep in subs:
                        new = pat.sub(rep, new)
                    if new != text:
                        edited.append((name, sum(len(p.findall(text)) for p, _ in subs)))
                        data = new.encode("utf-8")
            info = zipfile.ZipInfo("supplementary/" + name, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
            n += 1
        # list the paper/ folder as one entry, so that the zip does not change when only paper/ changes
        shown = sorted({"paper/ (the whole folder)" if e.startswith("paper/") else e for e in excluded})
        readme = README.format(n=n, excluded="\n".join(f"- `{e}`" for e in shown),
                               edited="\n".join(f"- `{e}`: {k} substitution(s)" for e, k in edited))
        info = zipfile.ZipInfo("supplementary/README_SUPPLEMENTARY.md", date_time=FIXED_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, readme.encode("utf-8"))
    return {"files": n + 1, "excluded": excluded, "edited": edited, "bytes": zip_path.stat().st_size}


README = """# Supplementary material (anonymized)

This is the paper's repository: the simulator and its tests, all training and evaluation code, every
configuration, the pre-registrations, the prediction entries, the run log, all reports and committed outputs,
the frozen checkpoint hashes, and the paper-materials pack (`results/paper/`: numbers, claims, timeline,
audit, figure scripts). {n} files.

Not included: the generated data arrays and the checkpoints, which are not in the repository (the data can
be regenerated from `configs/`; checksums are in each manifest), and these files:

{excluded}

(`paper/` holds the LaTeX source, whose preprint branch names the author; the drafts carry a byline.)

**Anonymization.** Text files were edited only by substitution: session scratchpad paths became
`<scratchpad>`, other home-directory paths `<home>`, the author's name, user names and e-mail address
`[anonymized]`, and the repository's folder name `anonymized-project`. Binary files are unchanged. Files
edited:

{edited}

Start with `README.md`, then `PREREGISTRATION.md`, `REPORT.md`, `PREREGISTRATION_v2.md`, `REPORT_v2.md`,
`PREREGISTRATION_v3.md`, `REPORT_v3.md` and `AUTOPSY.md`. `results/paper/NUMBERS.md` traces every number in
the paper to its source file and commit; commit hashes refer to the repository's history, which is not part
of this copy.
"""
