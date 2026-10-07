# Re:Cognize: 5-minute conference talk

Canva deck: <https://www.canva.com/d/5BlXqOtZsn5h6UH> ("Re:Cognize — NeurIPS 2026 talk (5 min)", 10 slides, 1920×1080).
Speaker notes in Canva carry a ~30 s script per slide (about 4:45 in total) and an animation cue.

The figure pages (question, binding, pricing) are copies of the native Canva figures from
"Re:Cast method diagrams", so every box and arrow is a separate, animatable element. The bar charts are
native Canva shapes drawn from the released results. `media/` holds MP4 versions of the two project-page
replays, which Canva plays automatically when the slide opens.

## Flow

| # | Slide | Message | Moving parts |
|---|---|---|---|
| 1 | Title | A reader knows a character a hundred pages later without a cast list | crop stream along the reading-order arrow |
| 2 | The question | Re-ID hands over the cast in advance; a reader never gets one | native teaser figure, three columns |
| 3 | The benchmark | One stream in reading order, four galleries (P1–P4) | `protocols-loop.mp4`, protocol cards, dataset stats |
| 4 | Finding one | Recognising is close to solved | P1 vs P2 bars, mAP and Rank-1 |
| 5 | Finding two | Knowing what to believe is not | static / own top-1 / true-label bars |
| 6 | One comparison | Δ = c · (p_eff − a⁺): commit or hold | equation, three term cards, commit/hold boxes |
| 7 | Re:Cast | Grow only where the page vouches | three change cards, `recast-loop.mp4` |
| 8 | Binding | With first-appearance seeds, the story carries the label forward | native binding figure (a)–(c) |
| 9 | Price before you commit | Any encoder can bind; Δ decides whether to trust it | native binder → decide pipeline |
| 10 | Takeaway | Re:Cognize measures whether a model can read along; Re:Cast is a cast that does | three rows, links |

## Animating in Canva

Canva (checked against the Canva Help Center) plays element animations when a slide appears; it cannot
yet trigger them on click or let you reorder them, and videos autoplay in Present mode. So:

- **Entrance builds**: select an element → Animate → pick an effect (Rise, Wipe, Pan, Typewriter for
  titles). Elements animate as the slide opens.
- **Step-by-step reveals** (e.g. slide 5: grey bars → orange → blue; slide 6: equation → cards → verdicts):
  duplicate the page, delete the not-yet-shown elements from the earlier copy, and set the transition
  between the copies to **Match and move**. Bars that exist on both pages glide; new ones fade in.
- **Charts that grow**: on the earlier copy, make each bar 1 px tall at the baseline; Match and move then
  animates them rising to full height.
- Keep page transitions short (0.5 s) so a 30 s slide is not spent waiting.

## Where the numbers come from

All numbers are in the paper (arXiv 2609.34032); chart values were regenerated from the released results
(`scripts/fetch_results.py`) with the repo's own generators.

| Slide | Numbers | Source |
|---|---|---|
| 3 | 4,058 crops / 70 characters / 8 series; 29,315 crops / 784 characters / 27 volumes; 5 backbones; 84 runs | §3, `docs/protocols.md`, `README.md` |
| 4 | P1 vs P2 (k = 1) mAP and Rank-1 per backbone, memory-block configuration | Fig. 3a (`paper/paper_figures.py`); the paper states these as 102–107 % of P1 mAP and 43–66 % of Rank-1 |
| 5 | P4 identity Rank-1: static, own top-1, true label; 62–89 % wrong additions; cosine 0.7 | Fig. 3b, §4, App. A.16–A.17 |
| 6 | p_eff vs a⁺: 39.1 vs 31.9 (+2.57), 54.4 vs 67.8 (−6.23); 87 % page-constraint precision | Table 4, §5 |
| 7 | +3.3 to +6.5 at k = 5 (28–41 % of oracle); 1.43 crops per seed, 90.6 % correct; 5/8 vs 7/8 replay | Table 5, §6, project page |
| 8 | +12.5 to +16.9 with first-appearance seeds (59–70 % of oracle); shuffled −5.6 to −20.4 | Table 6, §7 |
| 9 | 9 of 10 galleries gain; MagiV2 on Manga109 loses 7.4 | §7 |
