"""Check every entry of paper/refs.bib against its source, and write paper/refs_check.tsv.

Sources (named in the comment line above each bib entry):
- Crossref (publisher-deposited metadata) for entries with a DOI;
- the arXiv abstract page;
- PMLR and NeurIPS proceedings pages (citation_* meta tags), and the PMLR volume pages for edited volumes;
- the OpenReview API (title search, matched by note id) for ICLR, TMLR and OpenReview-only papers.

Compared: title (case and punctuation ignored), author or editor surnames in order, year, and volume,
number and pages wherever the source gives them. An entry is `verified` only if every compared field
agrees. Each row records the SHA-256 of the bib entry as checked, so that paper/build.py can refuse a
`verified` status for an entry edited afterwards.

Needs the network; run by hand, not by the build:

    python paper/tools/check_refs.py
"""
import csv
import datetime
import hashlib
import html
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BIB = REPO / "paper/refs.bib"
OUT = REPO / "paper/refs_check.tsv"
UA = {"User-Agent": "Mozilla/5.0 (reference check for a paper build)"}


def fetch(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < tries - 1:
                time.sleep(20 * (i + 1))
                continue
            raise


# ---- bib parsing ----
def parse_bib(text):
    entries = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),", text):
        start, i, depth = m.start(), m.end(), 1
        while depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        raw = text[start:i]
        fields = {}
        j = 0
        body = raw[m.end() - start:-1]
        for fm in re.finditer(r"(\w+)\s*=\s*\{", body):
            if fm.start() < j:
                continue
            k, d = fm.end(), 1
            while d:
                d += {"{": 1, "}": -1}.get(body[k], 0)
                k += 1
            fields[fm.group(1).lower()] = body[fm.end():k - 1]
            j = k
        cm = re.search(r"% source: ([^\n]*)\n\s*$", text[:start])
        entries[m.group(2).strip()] = {"type": m.group(1).lower(), "fields": fields, "raw": raw,
                                        "source": cm.group(1).strip() if cm else ""}
    return entries


ACCENTS = {'"': "̈", "'": "́", "~": "̃", "`": "̀", "^": "̂"}


def detex(s):
    s = re.sub(r"\{\\([\"'~`^])\{?\\?([A-Za-z])\}?\}",
               lambda m: unicodedata.normalize("NFC", m.group(2) + ACCENTS[m.group(1)]), s)
    return s.replace("{", "").replace("}", "").replace("\\", "")


def fold(s):
    s = unicodedata.normalize("NFKD", detex(s))
    return "".join(c for c in s if not unicodedata.combining(c)).replace("ı", "i").lower()


def norm_title(s):
    return " ".join(re.findall(r"[a-z0-9]+", fold(s)))


def surname(name):
    name = fold(name).strip()
    last = name.split(",")[0] if "," in name else name.split()[-1]
    return re.findall(r"[a-z\-]+", last)[-1] if re.findall(r"[a-z\-]+", last) else last


def bib_people(f):
    return [surname(a) for a in re.split(r"\s+and\s+", f)]


def pages(s):
    return re.sub(r"[^0-9]+", "-", s or "").strip("-")


# ---- sources ----
def src_crossref(doi):
    m = json.loads(fetch("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    yr = (m.get("published-print") or m.get("published") or m.get("issued"))["date-parts"][0][0]
    return {"title": (m.get("title") or [""])[0], "people": [a.get("family", "") for a in m.get("author", [])],
            "year": str(yr), "volume": m.get("volume"), "number": m.get("issue"),
            "pages": m.get("page") or m.get("article-number"), "venue": (m.get("container-title") or [""])[0]}


def meta_tags(page):
    out = {}
    for name, content in re.findall(r'<meta\s+name="(citation_[a-z_]+)"\s+content="([^"]*)"', page):
        out.setdefault(name, []).append(html.unescape(content))
    return out


def src_meta(url):
    t = meta_tags(fetch(url))
    date = (t.get("citation_publication_date") or t.get("citation_date") or [""])[0]
    first, last = (t.get("citation_firstpage") or [None])[0], (t.get("citation_lastpage") or [None])[0]
    return {"title": t["citation_title"][0], "people": t.get("citation_author", []), "year": date[:4],
            "volume": (t.get("citation_volume") or [None])[0], "number": None,
            "pages": f"{first}-{last}" if first and last else None}


def src_pmlr_volume(vol):
    page = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", fetch(f"https://proceedings.mlr.press/v{vol}/"))).split())
    eds = re.search(r"Volume Edited by:\s*(.*?)\s*Series Editors", page).group(1)
    title = re.search(r'"description":"(.*?) Held in', page).group(1)
    year = re.search(r"Machine Learning Research on \d+ \w+ (\d{4})", page).group(1)
    # the page lists the editors as one run of full names, so the bib's editors are compared as the same run
    return {"title": title, "people_run": eds, "year": year, "volume": str(vol), "number": None, "pages": None}


def src_arxiv(aid):
    r = src_meta(f"https://arxiv.org/abs/{aid}")
    r["volume"] = r["pages"] = None
    return r


def src_openreview(nid, title, api):
    q = urllib.parse.urlencode({"term": title, "type": "terms", "content": "all", "source": "all", "limit": 50})
    notes = json.loads(fetch(f"{api}/notes/search?{q}")).get("notes", [])
    n = next(n for n in notes if n.get("id") == nid)
    val = lambda x: x.get("value") if isinstance(x, dict) else x
    c = n["content"]
    stamp = n.get("pdate") or n.get("cdate") or n.get("tcdate")
    venue = val(c.get("venue")) or "OpenReview"
    year = re.search(r"(?:ICLR|TMLR)\s+(\d{4})", venue)
    year = year.group(1) if year else str(datetime.datetime.fromtimestamp(stamp / 1000, datetime.timezone.utc).year)
    return {"title": val(c["title"]), "people": val(c["authors"]), "year": year, "volume": None, "number": None,
            "pages": None, "venue": venue}


def source_for(key, e):
    s = e["source"]
    f = e["fields"]
    if s.startswith("Crossref"):
        return "https://doi.org/" + f["doi"], lambda: src_crossref(f["doi"])
    if s.startswith("arXiv"):
        aid = f["eprint"]
        return f"https://arxiv.org/abs/{aid}", lambda: src_arxiv(aid)
    if s.startswith("PMLR volume"):
        vol = f["volume"]
        return f"https://proceedings.mlr.press/v{vol}/", lambda: src_pmlr_volume(vol)
    if s.startswith("PMLR v"):
        url = "https://proceedings.mlr.press/" + s.split()[1] + ".html"
        return url, lambda: src_meta(url)
    if s.startswith("NeurIPS"):
        year, h = re.search(r"(\d{4}) hash (\w+)", s).groups()
        url = (f"https://proceedings.neurips.cc/paper/{year}/hash/{h}-Abstract.html" if year == "2021" else
               f"https://proceedings.neurips.cc/paper_files/paper/{year}/hash/{h}-Abstract"
               + ("-Conference" if int(year) >= 2022 else "") + ".html")
        return url, lambda: src_meta(url)
    if s.startswith("OpenReview"):
        nid = s.split()[1]
        api = "https://api2.openreview.net" if "TMLR" in s else "https://api.openreview.net"
        return f"https://openreview.net/forum?id={nid}", lambda: src_openreview(nid, detex(f["title"]), api)
    raise ValueError(f"{key}: no usable source line")


def compare(e, src):
    f = e["fields"]
    problems, checked = [], []
    bib_title = f.get("title", "")
    if norm_title(bib_title) != norm_title(src["title"]):
        problems.append(f"title: bib '{detex(bib_title)}' vs source '{src['title']}'")
    checked.append("title")
    who = f.get("author") or f.get("editor")
    if "people_run" in src:  # full names in one run: compare "First Last" names in order
        b = " ".join(" ".join(reversed([x.strip() for x in a.split(",", 1)])) for a in re.split(r"\s+and\s+", who))
        if " ".join(re.findall(r"[a-z]+", fold(b))) != " ".join(re.findall(r"[a-z]+", fold(src["people_run"]))):
            problems.append(f"people: bib '{fold(b)}' vs source '{fold(src['people_run'])}'")
    else:
        b, s = bib_people(who), [surname(p) for p in src["people"]]
        if b != s:
            problems.append(f"people: bib {b} vs source {s}")
    checked.append("authors" if "author" in f else "editors")
    if f.get("year") != src["year"]:
        problems.append(f"year: bib {f.get('year')} vs source {src['year']}")
    checked.append("year")
    for k in ("volume", "number", "pages"):
        if src.get(k) and f.get(k):
            if pages(f[k]) != pages(str(src[k])):
                problems.append(f"{k}: bib {f[k]} vs source {src[k]}")
            checked.append(k)
        elif src.get(k) and not f.get(k):
            problems.append(f"{k}: missing in bib (source {src[k]})")
    if src.get("venue", "").startswith("ICLR") and "Learning Representations" not in f.get("booktitle", ""):
        problems.append(f"venue: source {src['venue']}")
    if src.get("venue", "").startswith("Accepted by TMLR") and "Transactions on Machine Learning Research" not in f.get("journal", ""):
        problems.append(f"venue: source {src['venue']}")
    return problems, checked


def main():
    entries = parse_bib(BIB.read_text(encoding="utf-8"))
    rows = []
    for key, e in entries.items():
        url, get = source_for(key, e)
        try:
            src = get()
            problems, checked = compare(e, src)
            status = "verified" if not problems else "mismatch"
            detail = "; ".join(problems) or "all compared fields agree"
        except Exception as ex:  # noqa: BLE001
            status, detail, checked = "unreachable", f"{type(ex).__name__}: {ex}", []
        sha = hashlib.sha256(e["raw"].encode("utf-8")).hexdigest()[:16]
        rows.append({"key": key, "status": status, "source": url, "checked": ",".join(checked), "detail": detail,
                     "entry_sha256": sha, "date": datetime.date.today().isoformat()})
        print(f"{status:>11}  {key:<26} {detail[:110]}")
        time.sleep(2 if "openreview" not in url else 6)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    n = sum(r["status"] == "verified" for r in rows)
    print(f"check_refs: {n} of {len(rows)} verified; wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
