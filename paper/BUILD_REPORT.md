# Phase 3 build report (2026-11-03)

**Source:** `results/paper/DRAFT_v3.md` @ `e4c25f6`, with `results/paper/NUMBERS.md` §8 added in `bf15cee`.
No sealed or retired data was opened, and nothing was trained. The only data-free computation was the
decoder parameter count, now committed with its output (`results/paper/tools/param_count.*`).

## Outputs

| file | what |
|---|---|
| `out/anonymized.pdf` | TMLR submission build: "Anonymous authors", TMLR header; PDF info holds the title only, with no XMP and no dates. 28 pages. |
| `out/preprint.pdf` | Named preprint (`[preprint]` option): author block, no TMLR header; PDF info holds the title and author. 28 pages. |
| `out/supplementary_anonymized.zip` | Anonymized repository: every tracked file outside `paper/`, as at `bf15cee` (unchanged since). Not committed (see Open items). It rebuilds byte for byte from git HEAD (checked twice). |
| `main.tex`, `main_preprint.tex` | One source, two builds. `main.tex` is generated from the draft by `tools/md2tex.py`. |
| `refs.bib`, `refs_check.tsv` | References and the record of their source check. |
| `figures/` | Copies of `results/paper/figures/*.pdf` (byte-identical, checked by the build). |
| `build.py`, `tools/` | The build and its checks. |

**Rebuild:** `python paper/build.py`. It needs Tectonic (on PATH, or set `TECTONIC`) and `pypdf`. The
reference check `python paper/tools/check_refs.py` needs the network and is run by hand. The build refuses a
bib entry edited since its check.

**Engine:** both PDFs were compiled with Tectonic 0.17.0 (XeTeX), since no TeX was installed on this
machine. The source is ASCII and uses only standard packages, but it has not been compiled with pdfLaTeX,
which arXiv uses.

## Conversion: what is not word-for-word the draft

`main.tex` is the draft's text, mechanically converted. Formulas come from an explicit table, and every
entry must match the draft or the converter stops. The departures:

1. **Dropped:**
   - the byline;
   - the reference list and its [CHECK] note (the references come from `refs.bib`);
   - the author notes.
2. **Abstract:** the three paragraphs are joined into one, because the TMLR template asks for one
   paragraph. The text is unchanged.
3. **Figure 2 caption:** the file pointer "(the E4 panel is `figures/fig2b_e4.pdf`, Appendix B)" becomes a
   closing sentence: "The E4 panel is Figure B1 (Appendix B)."
4. **Figure 7 caption:** the file name becomes "Figure D1 (Appendix D)".
5. **Two appendix figure captions written for the build** (please approve or replace):
   - **Figure B1:** "E4 = a/b, the energy–enstrophy mix of the enstrophy-tracking direction (Section 5.2),
     for the plain and VICReg autoencoders of Figure 2 on the exploration set: seeds and seed mean, log axis.
     E4 = 1 means the direction tracks log Z₀ alone; the dashed line is the untrained control (E)."
   - **Figure D1:** "As Figure 7, for all 11 scored models: main seeds 100–104, τ = 0 seeds 200–202 and
     untrained seeds 300–302, decoded on the same exploration frame; ✓/✗ give each direction's verdict on
     both sealed sets (C)."
6. **§8.4 release placeholder:**
   - anonymized build: "An anonymized copy of the repository is provided as supplementary material."
   - preprint: a bold placeholder, "[Repository link to be added before posting.]". There is no git remote,
     so there was no URL to use.
7. **§8.5:** the reference check passed (below). The [CHECK] marker is removed and the sentence added: "…
   by the AI coding agent, which had not written the manuscript, and every reference was checked against
   its source."

## Number lint

- **Result:** 976 numeric literals in the title, abstract, body, tables, captions and appendices; **0 not
  traced**. They match `NUMBERS.md` digit for digit, with signs kept.
- **Allow-list:** `lint_allow.tsv` has 3 entries. Each names a committed file, and the lint checks that the
  file contains the token:
  - the years of the two NeurIPS pre-registration workshops (in `refs.bib`);
  - Figure 4's plotting floor 10⁻⁴ (in `results/paper/figures/README.md`).
- **Negative tests:** a planted typo (0.829 → 0.828), a flipped sign (−0.026 → +0.026) and a changed power
  of ten (3×10⁻³ → 4×10⁻³) were each caught.
- **Limitation:** the lint works on tokens. It finds a number absent from the pack, not a number quoted in
  the wrong place; context was checked in AUDIT_v1.
- **Not linted:** numbers drawn inside figures, such as Figure 5's shares. The lint reads only the text.

## References: 27 of 27 verified

`tools/check_refs.py` compared title, author or editor surnames in order, year, and volume, number and
pages against each source. The sources:
- Crossref, for 14 DOIs;
- arXiv;
- PMLR and NeurIPS proceedings pages;
- PMLR volume pages, for the two edited workshop volumes;
- the OpenReview API, for ICLR 2017/2021/2022, TMLR and LeCun's paper.

All entries agree; the details are in `refs_check.tsv`.

**Corrected against the draft's list** (most were the entries written from memory):
- Kraichnan: journal "The Physics of Fluids".
- Okubo: "Deep Sea Research and Oceanographic Abstracts".
- Piloto: issue 9, pages 1257–1267.
- Alet: pages 16384–16397; author "Josh" Tenenbaum, as on the proceedings page.
- Mialon: pages 28973–29004.
- Odena: Distill 1(10).
- Hewitt & Liang: pages 2733–2743.
- Assran: pages 15619–15629.
- Bardes 2024: TMLR, OpenReview `QaCCuDfBk2`; the author is "Mido Assran" on the TMLR record.
- LeCun: "version 0.9.2" left out. The OpenReview record does not show it; copies of the PDF do (2022-06-27).

**Claims about the cited works:** every claim the draft makes about a reference was checked against its
abstract or full text. That covers the audit's 11 "ext." rows plus the TMLR policy, and all hold but one.

| where | the draft says | the source says | proposed fix for DRAFT_v4 |
|---|---|---|---|
| §2 ¶3 | "Unsupervised disentanglement is impossible without inductive biases on the model **or** the data (Locatello et al., 2019)" | "fundamentally impossible without inductive biases on **both** the models and the data" (abstract) | "… without inductive biases on both the model and the data …" |

Also confirmed:
- **TMLR policy (§8.5):** "LLMs may be used as general-purpose assistive tools", and authors are "fully
  responsible" (TMLR editorial policies).
- **Pre-registration workshops:** both prefaces describe papers reviewed on a protocol, before any results
  exist.

## Anonymity check: 0 hits

**What is searched for.** The search strings are found at run time, not written into the code:
- the preprint author block;
- the git user name and email;
- the OS user and machine name;
- the home folder and the repository folder name;
- generic patterns for e-mail addresses, Windows and Unix home paths, and GitHub links.

**Anonymized PDF:** 0 hits. The search covered:
- page text, extracted with pypdf;
- the document info, which is the title only, with no XMP;
- every raw and decompressed stream.

All fonts are embedded.

**Supplementary zip:** 0 hits across 560 files. Each file's name and content was searched, `.gz` files were
decompressed first, and the zip comment was checked.
- **Excluded:** `paper/` (its preprint branch names the author) and `results/paper/DRAFT_v1–v3.md` (the
  byline).
- **Edited by substitution only:**
  - `results/LOG.md`: 120 substitutions, all scratchpad paths containing the user name;
  - one each in `README.md`, `PREREGISTRATION.md`, `REPORT.md` and `requirements.txt`: the project
    name → `anonymized-project`;
  - `results/paper/AUDIT_v1.md`: the byline mention.
- **A README at the root of the zip** lists these edits.

**Zip size and hash:** 82,225,490 bytes (78.4 MB); SHA-256 `36e8671b4a98edb268aa14aaea205abd58ecc15eb92a8411850da7fe4a0a76b3`.
Its README lists `paper/` as one excluded folder, so commits that touch only `paper/` leave the zip unchanged.

**PDF hashes:** the builds are deterministic; two rebuilds gave identical files.
- `anonymized.pdf`: `b88b8e38630b29e1a80317a0783f4c96a7af6a3c5c228e9bd915450bf05b1c8c`
- `preprint.pdf`: `674a6f1a8b959ba09e4abe7ca3fab92f7b1c3dc6c4f96c53da3cfe6a7a3c62bc`

**Preprint:** it identifies the author as intended (name and email).

## Length

| | page |
|---|---|
| Main text ends (References begin) | 20 (19.7 pages of main text) |
| Appendices begin | 23 |
| Total | 28 |

The main text is **over 12 pages**. It holds about 8,750 words of prose (about 11 pages) plus 7 figures and
7 tables. Removing the section float barriers changes nothing, so the length is in the content, not the
layout.

**Proposed moves** (not applied). The numbers are measured by rebuilding scratch variants that move each
block to the end of the appendix named; each row includes the moves above it:

| move | main text (pages) |
|---|---|
| as drafted | 19.7 |
| §5.1 (with Figure 2) → App. B | 19.4 |
| §6.6 → App. E | 19.4 |
| §8.1–8.3 (pre-registrations, deviations, compute table) → App. F | 18.3 |
| Figures 5 and 6 → App. C | 17.3 |
| §6.2's build-history paragraph and §6.3's last two rules → App. C | 16.3 |
| Table 2 and §4.5 → App. A | 16.3 |
| §7.4 → App. D | 16.0 |
| Figure 7 → App. D (beside Figure D1) | 14.7 |
| Figure 4 and Table 4 → App. C | 14.0 |
| Figures 1 and 3 → App. A and B (Tables 1 and 3 keep the numbers) | 12.7 |

**Reading:**
- **§5.1 and §6.6 hold about 1.2 pages of text,** measured from the heading positions. In the current
  layout, though, the page breaks are set by the figures, so moving them alone saves only 0.3 pages. Their
  text saving shows once figures move too.
- **Moves alone reach about 12.7 pages, and only by sending five of the seven main figures to the
  appendix.** 12 pages also needs about 500 words of prose cuts.
- **Alternatively, keep the paper long.** The draft's author note says a main text over 12 pages lengthens
  TMLR review.

## Layout

- **Overfull line:** one remains, 2.2 pt, in a Table 5 header; not visible.
- **Figure D1:** scaled to 0.86 × the text height, so that it fits on one page with its caption.
- **Float placement:** floats are kept within their section (`placeins`). Without this, the appendix
  headings separated from their tables.

## Open items for the author

1. **Preprint author block:** confirm or replace the name "Manvendra Modgil", the affiliation "Independent
   researcher" and the email. They are inferred from the git and account email; they are set in
   `tools/md2tex.py` (`AUTHOR`).
2. **Preprint repository URL (§8.4):** none exists yet. This is needed before posting to arXiv.
3. **The Locatello wording in §2** (above).
4. **The two appendix figure captions** (above).
5. **Length:** choose moves from the table, or accept a long submission.
6. **Commit the supplementary zip?** I left it out of git: it is 78 MB and rebuilds byte for byte from
   HEAD. It sits in `paper/out/`.
