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

**SHA-256:**
- `anonymized.pdf`: `b9e640ad0088f4b05ea6cb5ed5c2ae8bad4c2082d6aa1e77f3ad862b612010e7`
- `preprint.pdf`: `e69cfaadb15b69bafd7365ecaaf75129f013811bbbf0e0d0e73acefb9b377b61`
- `arxiv_source.tar.gz`: `63b305ffc7add525d2c17ad32cb3db5a99aaae3afde93751621f4aa289fee40a`
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

## Build-time moves (verbatim) and bridging text

**Moved unchanged:**
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

**One verbatim quirk:** the moved build-history paragraph ends "Each step is listed in Appendix C". It now
sits in Appendix C itself, so it refers to Table C1 from inside that appendix.

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
