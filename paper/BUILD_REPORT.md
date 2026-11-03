# Build report (2026-11-03, second build: DRAFT_v4 and the investigator's decisions)

**Source:** `results/paper/DRAFT_v4.md` @ `71c469d`. It is DRAFT_v3 with one change: in §2, "on the model or
the data" becomes "on both the models and the data" (Locatello et al., 2019). The pack is
`results/paper/NUMBERS.md` @ `bf15cee`. No sealed or retired data was opened, and nothing was trained.

## Outputs (`paper/out/`)

| file | what |
|---|---|
| `anonymized.pdf` | TMLR submission: "Anonymous authors", TMLR header. PDF info holds the title only, with no XMP. 29 pages. |
| `preprint.pdf` | Named preprint, compiled from `arxiv_source.tar.gz` exactly as arXiv does (below). PDF info holds the title and author. 30 pages. |
| `arxiv_source.tar.gz` | 13 files: `main.tex` (preprint switch set), `main.bbl`, `tmlr.sty`, `fancyhdr.sty`, 9 figure PDFs. 875 KB. |
| `supplementary_anonymized.zip` | For OpenReview; not committed. 564 files, 78.4 MB, built from git HEAD outside `paper/`. |
| `build_summary.txt` | The build's own log of every check. |
| `ARXIV_METADATA.md` | Title, plain-ASCII abstract (1,914 characters), comments field (30 pages, 9 figures, repository link) and categories (cs.LG; cross-lists cs.AI, physics.flu-dyn). |

**SHA-256:**
- `anonymized.pdf`: `752939f90f31e4fc101cf726d945394df0c72c35e1732fdee22ef9c23f87da10`
- `preprint.pdf`: `2ac2618d4048740d61a07e6e4e1ad0e1229c043123ea98ef422d3e683409aec1`
- `arxiv_source.tar.gz`: `20ff55dd2fe28cf123569ba2e7d5a5c209e640187d93f968719aa5417d80d1d3`
- `supplementary_anonymized.zip`: `65eecdc515d4b1c7540a33432398310039740eaf04e245215f2500b9deb16d7f`

Rebuilding gives byte-identical PDFs.

## Length

| | anonymized | preprint |
|---|---|---|
| Main text | **17.3 pages**; ends on page 18 | ends on page 18 |
| Appendices start | page 20 | page 21 |
| Total | 29 pages | 30 pages |

The decision expected about 16 pages. The difference is Figure 2, which stays in the main text by the same
decision. My earlier estimate had moved it out with §5.1; measured on this build, moving it would give 16.3
pages. Applying the float barriers only in the appendix would save 0.1 pages, so the layout is unchanged.

## Decisions applied

1. **Author block (preprint).** Copied from Paper 2's LaTeX source (`paper2_release/paper/main.tex`, the
   "Wall-Clock-Calibrated State Monitors" paper; the same block is in the GitHub repo
   `intervention-timing-saturation-trap`): Manvendra Modgil, Modint Intelligence,
   manvendramodgil.ai@gmail.com. Paper 1's source (`intervention_timing_paper.tex`) has the same name and
   email and no affiliation.
2. **Repository.** https://github.com/2025eb1100268-tech/readable-not-steerable, public, full history.
   - **Licences:** MIT for code (`LICENSE`), CC BY 4.0 for reports, results, figures and the paper
     (`LICENSE-CC-BY-4.0.txt`, the official legal code); the README says which applies to what.
   - **§8.4 of the preprint** links the repository and says the scored checkpoints of Studies 2 and 3 are
     attached to its v1.0 release.
   - **Checkpoints:** the 50 frozen checkpoints (39 + 11) total 504,932,071 bytes (481.5 MiB), under 2 GB;
     all 50 SHA-256 values match the frozen ones. See `results/paper/tools/checkpoint_manifest.txt`.
   - The anonymized build keeps "An anonymized copy of the repository is provided as supplementary
     material."
3. **Locatello wording:** applied as DRAFT_v4 (`71c469d`), the only change from v3.
4. **Captions B1 and D1:** both already end with a claim tag, (E) and (C). No change was needed.
5. **Length moves:** applied at build time (see below).
6. **Supplementary zip:** not committed.
7. **pdfLaTeX:** TeX Live 2026 is installed (portable, per user, at `C:\Users\Manav\texlive`; pdfTeX
   1.40.29). Docker isn't available on this machine, so the texlive/texlive image was not an option.
   - **Both builds use pdfLaTeX.** The generated `main.tex` and `refs.bib` are pure ASCII: Greek letters,
     sub- and superscripts, ✓/✗ and every other symbol are macros. The build checks this and fails
     otherwise.
   - **The preprint PDF is compiled as arXiv does.** The tarball is unpacked into an empty folder, and
     pdflatex runs until the log asks for no rerun (3 passes), with no BibTeX: the `.bbl` ships in the
     tarball. Both builds have no undefined references or citations, and one overfull line of 2.2 pt
     (not visible).
   - **Version caveat:** arXiv's TeX Live version may differ from 2026. The `.bbl` is plain natbib output,
     so it should not depend on that.

## Build-time moves, bridging text and build edit

**Moved unchanged** (except one sentence, removed by the build edit below):
- §5.1 → Appendix B.1. Figure 2 stays in the main text.
- §6.6 → Appendix E.1.
- §6.2's paragraph "This design was reached adaptively…" → Appendix C.
- §6.3's paragraphs "Latent orthogonality is not output independence…" and "A third rule…" → Appendix C.
- Figures 5 and 6 → Appendix C (now Figures C1 and C2; the main-text Figure 7 is now Figure 5).
- §8.1–8.3 → Appendix F.1–F.3 (Table 6 is now Table F2).

References to a moved section now read "Appendix" (for example, "Appendix E.1" in §6 and §7.3).

**Bridging text, the only new prose:**
- **§5, after the opening paragraph** (Figure 2 follows it): "These hypotheses came from an exploratory
  line on fresh data, summarized in Figure 2 and described in Appendix B.1, in which we call the pixel
  predictor the 'imaginer' and a JEPA trained at the same horizons the 'summarizer'."
- **§6.2, after the list:** "Appendix C describes how this design was reached, step by step."
- **§6.3, after "Loudness-null PCA":** "Two other shape rules failed: the leading principal component after
  projecting loudness out of the latent, and the directions of largest raw output sensitivity
  (Appendix C)."
- **§6.4, after "Decoder upsampling…":** "Appendix C shows the artefacts (Figure C1) and the dose curve
  (Figure C2)."
- **§8, opening paragraph:** "This section summarizes the record; Appendix F lists the deviations;" becomes
  "Appendix F gives the record and lists the deviations;".
- **§3.3 and §7.5:** "(Section 8)" becomes "(Appendix F.2)", where the criteria changes are now recorded.

**Build edit (2026-11-04):** the moved build-history paragraph ended "Each step is listed in Appendix C.",
and it now sits in Appendix C itself. That sentence is removed from both builds by `BUILD_EDITS` in
`tools/md2tex.py`; the draft keeps it.

Comparing the builds before and after the edit:
- **PDF text:** in both PDFs, the only difference in the full text is that sentence.
- **Pages:** only the page holding the paragraph renders differently (page 21 anonymized, page 22
  preprint); page counts are unchanged.
- **arXiv tarball:** only that one line of `main.tex` differs.

**Build edits (2026-11-04, second set).** The draft is unchanged; all three are applied by
`tools/md2tex.py`.
- **Abstract, preprint and arXiv builds only:** "will be released" becomes "are public", because the
  repository is public. The anonymized TMLR build keeps "will be released". The macro `\AbstractRelease` is
  set by the preprint switch.
- **Hyphenation:** `\hyphenation{VICReg}` stops the name breaking across lines; it broke as "VI-CReg" in
  Figure 2's caption.
- **Figure 5 caption:** the v3 dials figure, Figure 7 in the draft, now ends with the "(C)" tag.

Checked against the previous build:
- **`main.tex`:** differs only by these three edits.
- **Words:** with the running header and page numbers excluded, the anonymized PDF differs only by "(C)";
  the preprint differs by "(C)" and "will be released" → "are public".
- **"VI-CReg":** broken once before, never now.
- **Layout:** the hyphenation change alters one paragraph's line count, so text moves across page breaks
  on pages 9 and 11–13. The caption changes page 15, and the preprint abstract changes page 1. Page counts
  are unchanged (29 and 30), and so are the lints: 976 numeric literals, 0 untraced; ASCII only;
  anonymity 0 hits.

## Checks

- **References:** 27 of 27 verified against their sources (`refs_check.tsv`), unchanged since; the §8.5
  sentence is in.
- **Number lint:** 976 numeric literals, 0 not traced. The bridging text adds no numbers.
- **pdfLaTeX safety:** the source is ASCII only.
- **Anonymity:** 0 hits in the anonymized PDF and in the zip.
  - The search strings now also include the GitHub owner and the repository name.
  - The zip's `LICENSE` has the name replaced, like the other text files.

## Rebuild

`TEXLIVE_BIN=<TeX Live bin folder> python paper/build.py` (needs `pypdf`). The reference check
(`python paper/tools/check_refs.py`) needs the network and is run by hand.
