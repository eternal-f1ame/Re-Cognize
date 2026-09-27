# The Re:Cognize protocols, as implemented

Every reported number comes from the harness in `src/recognize/`. Its constants live in
`src/recognize/protocol_constants.py` and nowhere else, and every result JSON records them
(`recognize.provenance.stamp`).

## Data and splits

| Item | Rule |
|---|---|
| POPCharacters train / dev / test | 13 training series; development: Dragon Ball, Kuroko's Basketball; test: Bakuman, Demon Slayer, Dr Stone, Hunter x Hunter, Kagurabachi, Nisekoi, Oshi No Ko, Tokyo Ghoul (`configs/training/data_split.yaml`). Test series are read only by the final evaluation. |
| Evaluation set of a test series | All of its crops (4,058 over the 8 test series, 70 identities), with no split inside a series. |
| Crop padding | 0.10 of the box on every path. |
| Reading order | Natural sort of the page name (chapter, then page), then the annotation index (`recognize.data.page_order_key`). |
| Manga109 | Volume-disjoint split (`configs/training/data_split_manga109.yaml`). Evaluated zero-shot: checkpoints trained on POPCharacters, scored on the 27 held-out volumes. |
| Re:Verse | One series, Re:Zero, evaluated whole (1,825 crops, 12 identities), P1 only. |

## Backbones

| Key | Weights | Input | Normalisation | Native dim |
|---|---|---|---|---|
| transreid | released TransReID ViT-B/16 Re-ID checkpoint (Market-1501), strict load after key mapping | 256x128 | ImageNet | 768 |
| magiv2 | `ragavsachdeva/magiv2` crop encoder at a pinned revision, patch masking off (`mask_ratio = 0.0`, asserted at build) | 224x224 | ImageNet | 768 |
| magiv3 | `ragavsachdeva/magiv3` Florence-2 encoder at a pinned revision | 384x384 | ImageNet | 1024 |
| instructreid | released Instruct-ReID weights (visual encoder), strict load | 256x128 | ImageNet | 768 |
| reid5o | released ReID5o checkpoint, strict load | 384x128 | CLIP | 512 |

The BNNeck and the memory block operate at each backbone's native width, with no adapter. Every
backbone build is checked for determinism in evaluation mode (two forwards, identical output),
and every checkpoint load goes through `recognize.loading.load_state_dict_checked`.

## Protocols

All four use the BNNeck output, L2-normalised, and cosine similarity. Every stochastic choice
draws from seeds {0, 1, 2, 3, 4} (three seeds on Manga109). The retrieval protocols share one
metric function, `recognize.metrics.compute_retrieval_metrics` (exemplar-level AP, MRR, CMC at 1,
5 and 10), averaged over the queries of a series and then over series.

| Protocol | Definition |
|---|---|
| **P1**, closed set | Per identity, the gallery is max(1, floor(0.2 n)) random crops and the rest are queries. Identities with one crop are excluded and counted. |
| **P2**, seeded gallery | k in 1..5 seeds per identity: random (Seq-R) or the first k in reading order (Seq-T). An identity with at most k crops enters the gallery only, so the set of distractors is the same at every k. The queries are all crops that are not seeds. |
| **P4**, growing gallery | Starts as P2. Queries arrive in reading order and each is scored against the gallery as it stood before the query; then the query joins the buffer of its top-1 identity, with the seeds protected and B_max = 50 (swept over 0, 5, 10, 25, 50, 100 and unbounded). Reported as identity Rank-1. Update policies: `predicted` (the model's own top-1), `oracle` (the true identity) and `frozen` (no growth, which is P2). Also recorded: the wrong-append rate, the share of the grown gallery mislabelled at the end, and Rank-1 by quarter of the stream. |
| **P3**, emergence | The full stream in reading order into an empty gallery. Reference rule: join the most similar cluster if its normalised running centroid is within cosine tau_nov = 0.55, otherwise open a new cluster, with no bound on the count. Scored by the number of clusters, Purity, NMI, ARI and Hungarian accuracy. The alternative rules (variance-adaptive, density-aware, cohesion-relative, graph Louvain) are options of the same driver. P3 is a diagnostic: its rules are reference instantiations, not proposed methods. |
| With the memory block | Memory is initialised from the gallery crops only: each identity's prototypes are chosen by farthest-point sampling, from a starting crop drawn with a seed derived from the gallery, so the run is deterministic. Every other crop goes through two-pass routing: pass 1 searches all identities, pass 2 routes working memory by the pass-1 identity. No label is used after initialisation. P3 has no gallery to initialise from and runs on no-memory features. |
| Crop perturbations | Box displacement (shift10, shift20, shift30, tight07, loose13) and pixel corruption (jitter10, jitter20, blur2, blur4, occ15, occ30) are harness options with fixed definitions and seeds (`recognize.perturb`). |

## Training recipe

`src/recognize/recipe.py`: 200 epochs; AdamW, learning rate 1e-4, weight decay 1e-4; 5-epoch
linear warmup, then cosine decay to zero; FP16. PK sampling with P = 8 and K = 4, K_supp =
K_query = 2 (P = 4 for MagiV3); support and query never share a crop. ID-drop rho = 0.5 per sample
in pass 2, with the predicted identity. Prototype-loss temperature 0.15, triplet margin 0.3;
lambda_proto = 1, lambda_trip = 1, lambda_mem = 0.1, lambda_CE = 0.3 without the memory block and
0 with it. The memory modules train at the base learning rate. LoRA: r = 8, alpha = 16, last 4
layers, learning rate 1e-5. Memory block: K = 8 working-memory slots, S = 5 prototypes, 8 heads,
dropout 0.1.

Every configuration is trained three times (training seeds 0, 1, 2). The reported checkpoint is
the last epoch (`epoch_0200.pth`). A checkpoint is written every 10 epochs, together with a
development-split score; neither the development split nor that score enters a reported number.

## Provenance

Every result JSON carries the arguments and seeds, `PYTHONHASHSEED` (pinned to 0 by every launcher,
and checked), the MagiV2 mask ratio, B_max, the split, crop and identity counts, and the
checkpoint's SHA-256 and recorded configuration. The paper's tables and figures are
generated from these files by `paper/` (see `docs/reproducing.md`). A run of the harness also
records the git commit it ran from and whether the tree was modified. The published results omit
these two fields.
