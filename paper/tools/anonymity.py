"""Identity needles and the anonymity check for the anonymized PDF and the supplementary zip.

The needles are collected at run time, not written here: the author block of the named preprint
(tools/md2tex.py), the git user name and email, the OS user and machine name, the home directory, the
repository's folder name, and generic patterns for e-mail addresses, Windows home paths and GitHub links.
"""
import gzip
import io
import os
import re
import subprocess
import zipfile
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def needles():
    import md2tex  # noqa: E402  (same folder)
    words = set()
    name = md2tex.AUTHOR["author_name"]
    words.update(w for w in re.split(r"\W+", name) if len(w) > 2)
    words.add(md2tex.AUTHOR["author_email"])
    for key in ("user.name", "user.email"):
        out = subprocess.run(["git", "config", key], capture_output=True, text=True, cwd=REPO).stdout.strip()
        if out:
            words.add(out)
            words.add(out.split("@")[0])
    for env in ("USERNAME", "USER", "COMPUTERNAME", "HOSTNAME"):
        if os.environ.get(env):
            words.add(os.environ[env])
    words.add(Path.home().name)
    words.add(REPO.name)
    words.discard("")
    generic = {
        "e-mail address": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}",
        "Windows home path": r"[A-Za-z]:[\\/]+Users[\\/]",
        "Unix home path": r"/(?:home|Users)/[A-Za-z]",
        "GitHub link": r"github\.com/",
    }
    return sorted(words, key=str.lower), generic


EMAIL_OK = {"noreply@anthropic.com"}  # the AI-assistance attribution, not an author identity


def scan_text(text, label, words, generic, hits, emails=True):
    low = text.lower()
    squashed = re.sub(r"\s+", "", low)
    for w in words:
        n = low.count(w.lower()) + (squashed.count(w.lower().replace(" ", "")) if " " not in w else 0)
        if n:
            hits.append((label, w, n))
    for name, pat in generic.items():
        if name == "e-mail address" and not emails:
            continue  # binary data can spell an address-like string by chance
        found = [m for m in re.findall(pat, text) if m.lower() not in EMAIL_OK]
        if found:
            hits.append((label, name, len(found)))


def pdf_streams(b):
    out = []
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", b, re.S):
        try:
            out.append(zlib.decompress(m.group(1)))
        except Exception:  # noqa: BLE001
            pass
    return out


def check_pdf(path, words, generic):
    from pypdf import PdfReader
    hits = []
    b = Path(path).read_bytes()
    r = PdfReader(io.BytesIO(b))
    text = "\n".join(p.extract_text() or "" for p in r.pages)
    scan_text(text, "page text", words, generic, hits)
    meta = {k: str(v) for k, v in (r.metadata or {}).items()}
    scan_text(" ".join(f"{k} {v}" for k, v in meta.items()), "document info", words, generic, hits)
    raw = b"\n".join([b] + pdf_streams(b)).decode("latin-1")
    scan_text(raw, "raw and decompressed streams", words, generic, hits, emails=False)
    xmp = "/Metadata" in str(r.trailer["/Root"].keys())
    return {"pages": len(r.pages), "info": meta, "xmp": xmp, "hits": hits}


def check_zip(path, words, generic):
    hits = []
    with zipfile.ZipFile(path) as z:
        scan_text(z.comment.decode("latin-1"), "zip comment", words, generic, hits)
        for info in z.infolist():
            scan_text(info.filename, f"file name {info.filename}", words, generic, hits)
            b = z.read(info)
            blobs = [b]
            if info.filename.endswith(".gz"):
                blobs.append(gzip.decompress(b))
            if info.filename.endswith(".pdf"):
                blobs += pdf_streams(b)
            for x in blobs:
                try:
                    s, is_text = x.decode("utf-8"), True
                except UnicodeDecodeError:
                    s, is_text = x.decode("latin-1"), False
                scan_text(s, info.filename, words, generic, hits, emails=is_text)
        n = len(z.infolist())
    return {"files": n, "hits": hits}
