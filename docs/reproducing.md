# Reproducing the paper

Every number in the paper comes from one evaluation harness (`src/recognize/`) and a set of
analysis scripts (`analysis/`), and every table and figure is generated from their outputs by
`paper/`. This page gives the commands, in the order they depend on each other. With the results
archive (`python scripts/fetch_results.py`) you can skip straight to
[Tables and figures](#tables-and-figures): no GPU is needed for that step.

All commands run from the repository root with `src/` and `scripts/` on `PYTHONPATH` and the hash
seed pinned, which fixes the reading order of every stream:

```bash
export PYTHONPATH=$PWD/src:$PWD/scripts:$PYTHONPATH PYTHONHASHSEED=0
```

## 1. Training

The training campaign is `configs/runs.yaml` (the grid of four trained configurations per
backbone, the memory-block ablations, and the extra training runs of the finetuned and memory
cells) and `configs/runs_seeds.yaml` (training runs 1 and 2 of the two LoRA cells). The recipe is
`src/recognize/recipe.py`: 200 epochs, AdamW at 1e-4 with weight decay 1e-4, a 5-epoch warmup and
cosine decay, PK sampling with P = 8 and K = 4 (P = 4 for MagiV3), FP16. The reported checkpoint
is the last epoch, `checkpoints/<backbone>/<config>/seed<s>/final.pth`; `epoch_0200.pth` holds the
same weights, and each result file names the one it evaluated.

```bash
python scripts/train.py list                                  # every run in the manifest
python scripts/train.py run magiv2_memory_seed0               # one run, here
python scripts/train.py --manifest configs/runs_seeds.yaml list
python scripts/train.py sbatch --groups grid                  # the grid as a SLURM array
```

The 84 runs behind the reported numbers take about 60 GPU-hours (0.45 to 1.27 each).

## 2. Evaluation

`scripts/evaluate.py` scores one checkpoint, or one released backbone, on every protocol and
writes one JSON per series. The four groups of runs below produce `results/popcharacters/`,
`results/manga109/` and `results/reverse/`. `scripts/slurm/submit_eval.sh --groups popcharacters
manga109 reverse perturbations` submits all of them as one SLURM array; the manifest it builds holds
one row per command.

```bash
GRID="--protocols p1 p2 p3 p4 --seeds 0 1 2 3 4 --k 1 2 3 4 5 --b-max-sweep \
      --update-policies predicted oracle frozen \
      --p3-rules fixed variance-adaptive density-aware cohesion-relative graph-louvain"

# popcharacters: POPCharacters test series, every trained run and every released backbone
python scripts/evaluate.py --checkpoint checkpoints/magiv2/memory/seed0/final.pth $GRID \
    --out results/popcharacters/magiv2_memory_seed0
python scripts/evaluate.py --pretrained magiv2 $GRID --no-memory --out results/popcharacters/pretrained__magiv2

# manga109: Manga109, zero-shot, the grid runs of training run 0
python scripts/evaluate.py --checkpoint checkpoints/magiv2/memory/seed0/final.pth --dataset manga109 \
    --protocols p1 p2 p3 p4 --seeds 0 1 2 --k 1 3 5 --p3-rules fixed \
    --out results/manga109/magiv2_memory_seed0__manga109

# reverse: Re:Verse, P1
python scripts/evaluate.py --checkpoint checkpoints/magiv2/memory/seed0/final.pth --dataset reverse \
    --protocols p1 --seeds 0 1 2 3 4 --out results/reverse/magiv2_memory_seed0__reverse

# perturbations: crop perturbations (box: shift10 shift20 shift30 tight07 loose13; pixel: jitter10
#      jitter20 blur2 blur4 occ15 occ30), P1 and P2, then P4 under the same perturbation
python scripts/evaluate.py --checkpoint checkpoints/magiv2/finetuned/seed0/final.pth \
    --protocols p1 p2 --seeds 0 1 2 --k 1 --box-noise shift20 --out results/popcharacters/magiv2_finetuned_seed0__shift20
python scripts/evaluate.py --checkpoint checkpoints/magiv2/finetuned/seed0/final.pth --box-noise shift20 \
    --protocols p4 --k 1 --strategies random temporal --update-policies predicted oracle frozen \
    --out results/popcharacters/magiv2_finetuned_seed0__shift20__p4
```

P3 at the loose threshold (`results/p3tau80/`) is the same command with `--protocols p3 --tau 0.80`.

Without the memory block, a rerun reproduces the archived files up to floating-point differences
between devices (TransReID on CPU matches them to the sixth decimal). With the memory block, each
identity's prototypes grow from one starting crop drawn with a seed derived from the gallery, so
reruns agree with each other exactly; the archived memory-model files each hold a different draw of
that start. That moves the means over the eight test series by at most 0.07 mAP on P1 and P2 and
0.32 identity Rank-1 on P4 (TransReID with memory, training run 0), inside the spread between
training runs; a single series can move by up to 0.9 mAP, and by up to 6.6 identity Rank-1 on P4
under chronological seeding.

## 3. Analyses

Each script answers one question the paper asks of the harness, and writes one JSON under
`results/`. These are the commands that produced the files in the results archive. Page
groups (`results/panels*/`) come first, since most of the gallery analyses read them.
Manga109 runs over three series chunks are merged with `analysis/merge_chunks.py`.

### `analysis/detect_panels.py`

```bash
python analysis/detect_panels.py --device cuda --batch-size 8 --out results/panels
python analysis/detect_panels.py --device cuda --data-root Datasets/manga109 --out results/panels_manga109 --series "ARMS" "GakuenNoise" "HanzaiKousyouninMinegishiEitarou" "MagicianLoad" "OhWareraRettouSeitokai" "PrayerHaNemurenai" "TasogareTsushin" "TotteokiNoABC" "WarewareHaOniDearu"
python analysis/detect_panels.py --device cuda --data-root Datasets/manga109 --out results/panels_manga109 --series "BakuretsuKungFuGirl" "GarakutayaManta" "HinagikuKenzan" "MiraiSan" "PikaruGenkiDesu" "PsychoStaff" "TennenSenshiG" "ToutaMairimasu" "YukiNoFuruMachi"
python analysis/detect_panels.py --device cuda --data-root Datasets/manga109 --out results/panels_manga109 --series "BurariTessenTorimonocho" "GinNoChimera" "KyokugenCyclone" "OL_Lunch" "PlatinumJungle" "SyabondamaKieta" "TensiNoHaneToAkumaNoShippo" "TsubasaNoKioku" "YumeNoKayoiji"
python analysis/detect_panels.py --device cuda --data-root Datasets/manga109 --out results/panels_manga109 --series-file results/series/m109_chunk0.lst
python analysis/detect_panels.py --device cuda --data-root Datasets/manga109 --out results/panels_manga109 --series-file results/series/m109_chunk1.lst
python analysis/detect_panels.py --device cuda --data-root Datasets/manga109 --out results/panels_manga109 --series-file results/series/m109_chunk2.lst
python analysis/detect_panels.py --device cuda --match one_to_one --min-iou 0.3 --out results/panels_o2o
python analysis/detect_panels.py --device cuda --out results/panels --series "Dragon Ball" "Kuroko S Basketball"
python analysis/detect_panels.py --device cuda --out results/panels --series-file results/series/dev.lst
```

### `analysis/stage1_refine.py`

```bash
python analysis/stage1_refine.py --device cuda --ious 0.3 0.4 0.5 --out results/stage1.json
```

### `analysis/commit_condition.py`

```bash
python analysis/commit_condition.py --device cuda --k 5 --strategy random --backbones instructreid --out results/commit_pop_instructreid.json
python analysis/commit_condition.py --device cuda --k 5 --strategy random --backbones magiv2 --data-root Datasets/manga109 --panels results/panels_manga109 --series-file results/series/m109_all.lst --out results/commit_m109_magiv2.json
python analysis/commit_condition.py --device cuda --k 5 --strategy random --backbones magiv2 --out results/commit_pop_magiv2.json
python analysis/commit_condition.py --device cuda --k 5 --strategy random --backbones magiv3 --data-root Datasets/manga109 --panels results/panels_manga109 --series-file results/series/m109_all.lst --out results/commit_m109_magiv3.json
python analysis/commit_condition.py --device cuda --k 5 --strategy random --backbones magiv3 --out results/commit_pop_magiv3.json
python analysis/commit_condition.py --device cuda --k 5 --strategy random --backbones reid5o --out results/commit_pop_reid5o.json
python analysis/commit_condition.py --device cuda --k 5 --strategy random --backbones transreid --out results/commit_pop_transreid.json
```

### `analysis/prototype_gallery.py`

```bash
python analysis/prototype_gallery.py --device cuda --k 1 --out results/proto_k1.json
python analysis/prototype_gallery.py --device cuda --k 2 --backbones transreid instructreid reid5o --out results/proto_other_k2.json
python analysis/prototype_gallery.py --device cuda --k 2 --out results/proto_k2.json
python analysis/prototype_gallery.py --device cuda --k 3 --backbones transreid instructreid reid5o --out results/proto_other_k3.json
python analysis/prototype_gallery.py --device cuda --k 3 --out results/proto_k3.json
python analysis/prototype_gallery.py --device cuda --k 5 --backbones magiv2 --data-root Datasets/manga109 --panels results/panels_manga109 --series-file results/series/m109_chunk0.lst --out results/proto_m109_k5_magiv2c0.json
python analysis/prototype_gallery.py --device cuda --k 5 --backbones magiv2 --data-root Datasets/manga109 --panels results/panels_manga109 --series-file results/series/m109_chunk1.lst --out results/proto_m109_k5_magiv2c1.json
python analysis/prototype_gallery.py --device cuda --k 5 --backbones magiv2 --data-root Datasets/manga109 --panels results/panels_manga109 --series-file results/series/m109_chunk2.lst --out results/proto_m109_k5_magiv2c2.json
python analysis/prototype_gallery.py --device cuda --k 5 --backbones magiv3 --data-root Datasets/manga109 --panels results/panels_manga109 --series-file results/series/m109_chunk0.lst --out results/proto_m109_k5_magiv3c0.json
python analysis/prototype_gallery.py --device cuda --k 5 --backbones magiv3 --data-root Datasets/manga109 --panels results/panels_manga109 --series-file results/series/m109_chunk1.lst --out results/proto_m109_k5_magiv3c1.json
python analysis/prototype_gallery.py --device cuda --k 5 --backbones magiv3 --data-root Datasets/manga109 --panels results/panels_manga109 --series-file results/series/m109_chunk2.lst --out results/proto_m109_k5_magiv3c2.json
python analysis/prototype_gallery.py --device cuda --k 5 --backbones transreid instructreid reid5o --out results/proto_other_k5.json
python analysis/prototype_gallery.py --device cuda --k 5 --out results/proto_k5.json
python analysis/prototype_gallery.py --device cuda --k 5 --strategy random --backbones magiv2 magiv3 --series-file results/series/dev.lst --out results/proto_dev_k5.json
python analysis/prototype_gallery.py --device cuda --k 5 --strategy temporal --backbones magiv2 --data-root Datasets/manga109 --panels results/panels_manga109 --series-file results/series/m109_chunk0.lst --out results/proto_m109_t5_magiv2c0.json
python analysis/prototype_gallery.py --device cuda --k 5 --strategy temporal --backbones magiv2 --data-root Datasets/manga109 --panels results/panels_manga109 --series-file results/series/m109_chunk1.lst --out results/proto_m109_t5_magiv2c1.json
python analysis/prototype_gallery.py --device cuda --k 5 --strategy temporal --backbones magiv2 --data-root Datasets/manga109 --panels results/panels_manga109 --series-file results/series/m109_chunk2.lst --out results/proto_m109_t5_magiv2c2.json
python analysis/prototype_gallery.py --device cuda --k 5 --strategy temporal --backbones magiv2 magiv3 --series-file results/series/dev.lst --out results/proto_dev_t5.json
python analysis/prototype_gallery.py --device cuda --k 5 --strategy temporal --backbones magiv3 --data-root Datasets/manga109 --panels results/panels_manga109 --series-file results/series/m109_chunk0.lst --out results/proto_m109_t5_magiv3c0.json
python analysis/prototype_gallery.py --device cuda --k 5 --strategy temporal --backbones magiv3 --data-root Datasets/manga109 --panels results/panels_manga109 --series-file results/series/m109_chunk1.lst --out results/proto_m109_t5_magiv3c1.json
python analysis/prototype_gallery.py --device cuda --k 5 --strategy temporal --backbones magiv3 --data-root Datasets/manga109 --panels results/panels_manga109 --series-file results/series/m109_chunk2.lst --out results/proto_m109_t5_magiv3c2.json
python analysis/prototype_gallery.py --device cuda --k 5 --strategy temporal --backbones transreid instructreid reid5o --out results/proto_other_t5.json
python analysis/prototype_gallery.py --device cuda --k 5 --strategy temporal --out results/proto_t5.json
```

### `analysis/seed_expansion.py`

```bash
python analysis/seed_expansion.py --device cuda --k 1 --out results/expand_k1.json
python analysis/seed_expansion.py --device cuda --k 2 --out results/expand_k2.json
python analysis/seed_expansion.py --device cuda --k 5 --out results/expand_k5.json
python analysis/seed_expansion.py --device cuda --k 5 --strategy temporal --out results/expand_t5.json
```

### `analysis/mustlink_append.py`

```bash
python analysis/mustlink_append.py --device cuda --k 5 --out results/mustlink_k5.json
python analysis/mustlink_append.py --device cuda --out results/mustlink.json
```

### `analysis/binder_calibration.py`

```bash
python analysis/binder_calibration.py --device cuda --backbones magiv3 --out results/binder_cal_m3.json
python analysis/binder_calibration.py --device cuda --backbones transreid instructreid reid5o magiv2 --out results/binder_cal_small.json
```

### `analysis/cross_page_link.py`

```bash
python analysis/cross_page_link.py --device cuda --chunk 4 16 48 --out results/cross_page_link.json
```

### `analysis/stage2_merge.py`

```bash
python analysis/stage2_merge.py --device cuda --panels results/panels_o2o --taus 0.5 0.7 0.9 --out results/stage2_reach.json
```

### `analysis/transport_append.py`

```bash
python analysis/transport_append.py --device cuda --panels results/panels_manga109 --data-root Datasets/manga109 --series-file results/series/m109_all.lst --tau 0.5 --stage1 self --strategies temporal --seeds 0 1 2 --backbones instructreid --out results/transport_m109_instructreid.json
python analysis/transport_append.py --device cuda --panels results/panels_manga109 --data-root Datasets/manga109 --series-file results/series/m109_all.lst --tau 0.5 --stage1 self --strategies temporal --seeds 0 1 2 --backbones magiv2 --out results/transport_m109_magiv2.json
python analysis/transport_append.py --device cuda --panels results/panels_manga109 --data-root Datasets/manga109 --series-file results/series/m109_all.lst --tau 0.5 --stage1 self --strategies temporal --seeds 0 1 2 --backbones magiv3 --out results/transport_m109_magiv3.json
python analysis/transport_append.py --device cuda --panels results/panels_manga109 --data-root Datasets/manga109 --series-file results/series/m109_all.lst --tau 0.5 --stage1 self --strategies temporal --seeds 0 1 2 --backbones reid5o --out results/transport_m109_reid5o.json
python analysis/transport_append.py --device cuda --panels results/panels_manga109 --data-root Datasets/manga109 --series-file results/series/m109_all.lst --tau 0.5 --stage1 self --strategies temporal --seeds 0 1 2 --backbones transreid --out results/transport_m109_transreid.json
python analysis/transport_append.py --device cuda --panels results/panels_o2o --backbones instructreid --link-source self --stage1 self --tau 0.35 --stage1-tau 0.35 --strategies temporal --out results/curve_instructreid.json
python analysis/transport_append.py --device cuda --panels results/panels_o2o --backbones instructreid --link-source self --stage1 self --taus 0.25 0.35 0.45 0.55 0.65 0.75 0.85 --strategies temporal --out results/exact_instructreid.json
python analysis/transport_append.py --device cuda --panels results/panels_o2o --backbones magiv2 --link-source self --stage1 self --tau 0.55 --stage1-tau 0.55 --strategies temporal --out results/curve_magiv2.json
python analysis/transport_append.py --device cuda --panels results/panels_o2o --backbones magiv2 --link-source self --stage1 self --taus 0.25 0.35 0.45 0.55 0.65 0.75 0.85 --strategies temporal --out results/exact_magiv2.json
python analysis/transport_append.py --device cuda --panels results/panels_o2o --backbones magiv3 --link-source self --stage1 self --tau 0.50 --stage1-tau 0.50 --strategies temporal --out results/curve_magiv3.json
python analysis/transport_append.py --device cuda --panels results/panels_o2o --backbones magiv3 --link-source self --stage1 self --taus 0.25 0.35 0.45 0.55 0.65 0.75 0.85 --strategies temporal --out results/exact_magiv3.json
python analysis/transport_append.py --device cuda --panels results/panels_o2o --backbones reid5o --link-source self --stage1 self --tau 0.40 --stage1-tau 0.40 --strategies temporal --out results/curve_reid5o.json
python analysis/transport_append.py --device cuda --panels results/panels_o2o --backbones reid5o --link-source self --stage1 self --taus 0.25 0.35 0.45 0.55 0.65 0.75 0.85 --strategies temporal --out results/exact_reid5o.json
python analysis/transport_append.py --device cuda --panels results/panels_o2o --backbones transreid --link-source self --stage1 self --tau 0.60 --stage1-tau 0.60 --strategies temporal --out results/curve_transreid.json
python analysis/transport_append.py --device cuda --panels results/panels_o2o --backbones transreid --link-source self --stage1 self --taus 0.25 0.35 0.45 0.55 0.65 0.75 0.85 --strategies temporal --out results/exact_transreid.json
python analysis/transport_append.py --device cuda --panels results/panels_o2o --tau 0.5 --link-source self --stage1 self --strategies temporal --out results/transport_self.json
python analysis/transport_append.py --device cuda --panels results/panels_o2o --tau 0.5 --link-source self --strategies temporal --out results/transport_link_self.json
python analysis/transport_append.py --device cuda --panels results/panels_o2o --tau 0.5 --out results/transport.json
python analysis/transport_append.py --device cuda --panels results/panels_o2o --tau 0.5 --stage1 self --strategies temporal --out results/transport_stage1_self.json
```

### `analysis/live_cast.py`

```bash
python analysis/live_cast.py --device cuda --k 5 --strategy random --backbones instructreid --data-root Datasets/manga109 --series-file results/series/m109_all.lst --out results/live_m109_k5_instructreid.json
python analysis/live_cast.py --device cuda --k 5 --strategy random --backbones instructreid --out results/live_pop_k5_instructreid.json
python analysis/live_cast.py --device cuda --k 5 --strategy random --backbones magiv2 --data-root Datasets/manga109 --series-file results/series/m109_all.lst --out results/live_m109_k5_magiv2.json
python analysis/live_cast.py --device cuda --k 5 --strategy random --backbones magiv2 --out results/live_pop_k5_magiv2.json
python analysis/live_cast.py --device cuda --k 5 --strategy random --backbones magiv2 magiv3 --series-file results/series/dev.lst --out results/live_dev_k5.json
python analysis/live_cast.py --device cuda --k 5 --strategy random --backbones magiv3 --data-root Datasets/manga109 --series-file results/series/m109_all.lst --out results/live_m109_k5_magiv3.json
python analysis/live_cast.py --device cuda --k 5 --strategy random --backbones magiv3 --out results/live_pop_k5_magiv3.json
python analysis/live_cast.py --device cuda --k 5 --strategy random --backbones reid5o --data-root Datasets/manga109 --series-file results/series/m109_all.lst --out results/live_m109_k5_reid5o.json
python analysis/live_cast.py --device cuda --k 5 --strategy random --backbones reid5o --out results/live_pop_k5_reid5o.json
python analysis/live_cast.py --device cuda --k 5 --strategy random --backbones transreid --data-root Datasets/manga109 --series-file results/series/m109_all.lst --out results/live_m109_k5_transreid.json
python analysis/live_cast.py --device cuda --k 5 --strategy random --backbones transreid --out results/live_pop_k5_transreid.json
python analysis/live_cast.py --device cuda --k 5 --strategy temporal --backbones instructreid --data-root Datasets/manga109 --series-file results/series/m109_all.lst --out results/live_m109_t5_instructreid.json
python analysis/live_cast.py --device cuda --k 5 --strategy temporal --backbones instructreid --out results/live_pop_t5_instructreid.json
python analysis/live_cast.py --device cuda --k 5 --strategy temporal --backbones magiv2 --data-root Datasets/manga109 --series-file results/series/m109_all.lst --out results/live_m109_t5_magiv2.json
python analysis/live_cast.py --device cuda --k 5 --strategy temporal --backbones magiv2 --out results/live_pop_t5_magiv2.json
python analysis/live_cast.py --device cuda --k 5 --strategy temporal --backbones magiv2 magiv3 --series-file results/series/dev.lst --out results/live_dev_t5.json
python analysis/live_cast.py --device cuda --k 5 --strategy temporal --backbones magiv3 --data-root Datasets/manga109 --series-file results/series/m109_all.lst --out results/live_m109_t5_magiv3.json
python analysis/live_cast.py --device cuda --k 5 --strategy temporal --backbones magiv3 --out results/live_pop_t5_magiv3.json
python analysis/live_cast.py --device cuda --k 5 --strategy temporal --backbones reid5o --data-root Datasets/manga109 --series-file results/series/m109_all.lst --out results/live_m109_t5_reid5o.json
python analysis/live_cast.py --device cuda --k 5 --strategy temporal --backbones reid5o --out results/live_pop_t5_reid5o.json
python analysis/live_cast.py --device cuda --k 5 --strategy temporal --backbones transreid --data-root Datasets/manga109 --series-file results/series/m109_all.lst --out results/live_m109_t5_transreid.json
python analysis/live_cast.py --device cuda --k 5 --strategy temporal --backbones transreid --out results/live_pop_t5_transreid.json
```

### `analysis/cluster_pool.py`

```bash
python analysis/cluster_pool.py --device cuda --k 5 --strategy random --backbones instructreid --data-root Datasets/manga109 --series-file results/series/m109_all.lst --out results/pool_m109_k5_instructreid.json
python analysis/cluster_pool.py --device cuda --k 5 --strategy random --backbones instructreid --out results/pool_pop_k5_instructreid.json
python analysis/cluster_pool.py --device cuda --k 5 --strategy random --backbones magiv2 --data-root Datasets/manga109 --series-file results/series/m109_all.lst --out results/pool_m109_k5_magiv2.json
python analysis/cluster_pool.py --device cuda --k 5 --strategy random --backbones magiv2 --out results/pool_pop_k5_magiv2.json
python analysis/cluster_pool.py --device cuda --k 5 --strategy random --backbones magiv2 magiv3 --series-file results/series/dev.lst --out results/pool_dev_k5.json
python analysis/cluster_pool.py --device cuda --k 5 --strategy random --backbones magiv3 --data-root Datasets/manga109 --series-file results/series/m109_all.lst --out results/pool_m109_k5_magiv3.json
python analysis/cluster_pool.py --device cuda --k 5 --strategy random --backbones magiv3 --out results/pool_pop_k5_magiv3.json
python analysis/cluster_pool.py --device cuda --k 5 --strategy random --backbones reid5o --data-root Datasets/manga109 --series-file results/series/m109_all.lst --out results/pool_m109_k5_reid5o.json
python analysis/cluster_pool.py --device cuda --k 5 --strategy random --backbones reid5o --out results/pool_pop_k5_reid5o.json
python analysis/cluster_pool.py --device cuda --k 5 --strategy random --backbones transreid --data-root Datasets/manga109 --series-file results/series/m109_all.lst --out results/pool_m109_k5_transreid.json
python analysis/cluster_pool.py --device cuda --k 5 --strategy random --backbones transreid --out results/pool_pop_k5_transreid.json
python analysis/cluster_pool.py --device cuda --k 5 --strategy temporal --backbones instructreid --data-root Datasets/manga109 --series-file results/series/m109_all.lst --out results/pool_m109_t5_instructreid.json
python analysis/cluster_pool.py --device cuda --k 5 --strategy temporal --backbones instructreid --out results/pool_pop_t5_instructreid.json
python analysis/cluster_pool.py --device cuda --k 5 --strategy temporal --backbones magiv2 --data-root Datasets/manga109 --series-file results/series/m109_all.lst --out results/pool_m109_t5_magiv2.json
python analysis/cluster_pool.py --device cuda --k 5 --strategy temporal --backbones magiv2 --out results/pool_pop_t5_magiv2.json
python analysis/cluster_pool.py --device cuda --k 5 --strategy temporal --backbones magiv2 magiv3 --series-file results/series/dev.lst --out results/pool_dev_t5.json
python analysis/cluster_pool.py --device cuda --k 5 --strategy temporal --backbones magiv3 --data-root Datasets/manga109 --series-file results/series/m109_all.lst --out results/pool_m109_t5_magiv3.json
python analysis/cluster_pool.py --device cuda --k 5 --strategy temporal --backbones magiv3 --out results/pool_pop_t5_magiv3.json
python analysis/cluster_pool.py --device cuda --k 5 --strategy temporal --backbones reid5o --data-root Datasets/manga109 --series-file results/series/m109_all.lst --out results/pool_m109_t5_reid5o.json
python analysis/cluster_pool.py --device cuda --k 5 --strategy temporal --backbones reid5o --out results/pool_pop_t5_reid5o.json
python analysis/cluster_pool.py --device cuda --k 5 --strategy temporal --backbones transreid --data-root Datasets/manga109 --series-file results/series/m109_all.lst --out results/pool_m109_t5_transreid.json
python analysis/cluster_pool.py --device cuda --k 5 --strategy temporal --backbones transreid --out results/pool_pop_t5_transreid.json
```

### `analysis/forward_commit.py`

```bash
python analysis/forward_commit.py --device cuda --backbones magiv2 transreid instructreid reid5o --out results/forward_commit_small.json
python analysis/forward_commit.py --device cuda --backbones magiv3 --out results/forward_commit_magiv3.json
```

### `analysis/rolling_gallery.py`

```bash
python analysis/rolling_gallery.py --device cuda --k 1 --out results/rolling_k1.json
python analysis/rolling_gallery.py --device cuda --k 5 --backbones transreid instructreid reid5o --out results/rolling_other_k5.json
python analysis/rolling_gallery.py --device cuda --k 5 --out results/rolling_k5.json
python analysis/rolling_gallery.py --device cuda --k 5 --strategy temporal --backbones transreid instructreid reid5o --out results/rolling_other_t5.json
python analysis/rolling_gallery.py --device cuda --k 5 --strategy temporal --out results/rolling_t5.json
```

### `analysis/seed_distance.py`

```bash
python analysis/seed_distance.py --device cuda --backbones magiv2 transreid instructreid reid5o --out results/seed_distance_small.json
python analysis/seed_distance.py --device cuda --backbones magiv3 --out results/seed_distance_magiv3.json
```

### `analysis/append_gate.py`

```bash
python analysis/append_gate.py --device cuda --backbones magiv2 magiv3 --out results/append_gate_recip.json
python analysis/append_gate.py --device cuda --backbones transreid magiv2 --out results/append_gate.json
```

### `analysis/nfc.py`

```bash
python analysis/nfc.py --device cuda --out results/nfc.json
```

### `analysis/dialogue_names.py`

```bash
python analysis/dialogue_names.py --device cuda --min-iou 0.3 --fuzzy --out results/names_iou30_fuzzy.json
python analysis/dialogue_names.py --device cuda --min-iou 0.3 --out results/names_iou30_exact.json
python analysis/dialogue_names.py --device cuda --min-iou 0.5 --fuzzy --out results/names_iou50_fuzzy.json
python analysis/dialogue_names.py --device cuda --out results/names.json
python analysis/dialogue_names.py --device cuda --series-file results/series/dev.lst --out results/names_dev.json
```

### `analysis/name_cast.py`

```bash
python analysis/name_cast.py --device cuda --k 5 --strategy random --backbones instructreid --names results/names.json --out results/nc_pop_k5_instructreid.json
python analysis/name_cast.py --device cuda --k 5 --strategy random --backbones magiv2 --names results/names.json --out results/nc_pop_k5_magiv2.json
python analysis/name_cast.py --device cuda --k 5 --strategy random --backbones magiv2 magiv3 --names results/names_dev.json --series-file results/series/dev.lst --out results/nc_dev_k5.json
python analysis/name_cast.py --device cuda --k 5 --strategy random --backbones magiv3 --names results/names.json --out results/nc_pop_k5_magiv3.json
python analysis/name_cast.py --device cuda --k 5 --strategy random --backbones reid5o --names results/names.json --out results/nc_pop_k5_reid5o.json
python analysis/name_cast.py --device cuda --k 5 --strategy random --backbones transreid --names results/names.json --out results/nc_pop_k5_transreid.json
python analysis/name_cast.py --device cuda --k 5 --strategy temporal --backbones instructreid --names results/names.json --out results/nc_pop_t5_instructreid.json
python analysis/name_cast.py --device cuda --k 5 --strategy temporal --backbones magiv2 --names results/names.json --out results/nc_pop_t5_magiv2.json
python analysis/name_cast.py --device cuda --k 5 --strategy temporal --backbones magiv2 magiv3 --names results/names_dev.json --series-file results/series/dev.lst --out results/nc_dev_t5.json
python analysis/name_cast.py --device cuda --k 5 --strategy temporal --backbones magiv3 --names results/names.json --out results/nc_pop_t5_magiv3.json
python analysis/name_cast.py --device cuda --k 5 --strategy temporal --backbones reid5o --names results/names.json --out results/nc_pop_t5_reid5o.json
python analysis/name_cast.py --device cuda --k 5 --strategy temporal --backbones transreid --names results/names.json --out results/nc_pop_t5_transreid.json
```

### `analysis/speaker_append.py`

```bash
python analysis/speaker_append.py --device cuda --calls results/names.json --backbones magiv3 --out results/speaker_append_magiv3.json
python analysis/speaker_append.py --device cuda --calls results/names.json --backbones transreid instructreid reid5o magiv2 --out results/speaker_append_small.json
python analysis/speaker_append.py --device cuda --calls results/names.json --panels results/panels --backbones magiv3 --out results/speaker_linked_magiv3.json
python analysis/speaker_append.py --device cuda --calls results/names.json --panels results/panels --backbones transreid instructreid reid5o magiv2 --out results/speaker_linked_small.json
```

### `analysis/p3_rule_sweep.py`

```bash
python analysis/p3_rule_sweep.py --device cuda --backbones magiv2 transreid --out results/p3_sweep.json
```

### `analysis/p3_tau.py`

```bash
python analysis/p3_tau.py --device cuda --backbones transreid magiv2 magiv3 instructreid reid5o --taus 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 --out results/p3_tau_sweep.json
python analysis/p3_tau.py --device cuda --out results/p3_tau.json
```

### `analysis/latency.py`

```bash
python analysis/latency.py --device cuda --out results/latency.json
```
### Reports read from the outputs above

```bash
python analysis/chance_baselines.py --draws 200      # chance levels of every protocol (recorded output in the archive)
python analysis/metric_floors.py                     # per-metric spread between training runs
python analysis/seed_distance_report.py results/seed_distance*.json --per-backbone
python analysis/forward_commit_report.py results/forward_commit_*.json
python analysis/merge_chunks.py results/proto_m109_k5_magiv2c{0,1,2}.json \
    results/proto_m109_k5_magiv3c{0,1,2}.json --out results/proto_m109_k5.json
python scripts/report/robustness.py                  # crop-perturbation tables
python scripts/report/timing.py                      # training cost per run
```

## Tables and figures

The generators write LaTeX tables to `paper/generated/tables/` and PDF figures to
`paper/generated/figures/`. `paper/figure_audit.py --figdir paper/generated/figures` checks every
figure's text at the size the paper prints it.

| In the paper | Source |
|---|---|
| Table 2, cross-protocol summary | `python paper/paper_tables.py --table cross --out paper/generated/tables/popcharacters_eval_scores.tex` |
| Table 3, P3 on full streams | `results/p3_tau.json` (fixed rule at 0.55 and 0.80) and the `nmi` fields of `results/popcharacters/*_finetuned_seed0/` and `results/p3tau80/` |
| Table 4, the commit condition | `python paper/paper_tables_recast.py --table commit --out paper/generated/tables/commit_condition.tex` |
| Table 5, Re:Cast one change at a time | `python paper/paper_tables_recast.py --table recast --out paper/generated/tables/recast_combined.tex` |
| Table 6, two-stage binding | `python paper/paper_tables_recast.py --table transport --out paper/generated/tables/transport_results.tex` |
| Figure 2, the four protocols | `python paper/protocols_figure.py` (draws crops from the datasets) |
| Figure 3 (a)-(c) | `python paper/paper_figures.py` |
| Figures 3 (d), 4 and 6 | `python paper/recast_figures.py` (Figure 4 draws crops from the datasets) |
| Figure 5, binding | layout drawn by hand; `paper/recast_figure_images.py` places the Re:Verse crops |
| Figures 7 and 8, the memory block | `python paper/mecha_figures.py` |
| Table 9, computational cost | `python paper/paper_tables.py --table cost` (from `analysis/latency.py`) |
| Table 10, full P1 grid | `python paper/paper_tables.py --table p1grid` |
| Table 11, per-manga results | `python paper/paper_tables.py --table permanga` |
| Table 12, transfer summary | `python paper/paper_tables_recast.py --table crosscorpus` |
| Table 13, Manga109 | `python paper/paper_tables.py --table manga109` |
| Table 14, Re:Verse | `results/reverse/` (the vision-language rows are those published with Re:Verse) |
| Tables 15 and 16, P3 rules and backbones | the `p3` fields of `results/popcharacters/*_seed0/` |
| Table 17, P4 decomposition | `python paper/paper_tables.py --table p4decomp` |
| Figure 9, P4 over the stream | `python paper/p4_drift.py` |
| Table 18, buffer cap | the `p4[...][b_max]` fields of the grid runs (`--b-max-sweep`) |
| Tables 19 and 20, crop perturbations | `python scripts/report/robustness.py` |
| Table 21, memory-block ablation | `python paper/paper_tables.py --table ablation` |
| Figure 10, P3 threshold | `python paper/threshold_sensitivity.py` (from `analysis/p3_tau.py` over 0.30 to 0.85) |
| Table 22, Figures 11 and 12, corpora | `python paper/dataset_stats.py --figures-only` (needs the datasets) |

Figure 1 and Tables 1, 7 and 8 are written or drawn by hand. `python paper/paper_tables.py --check`
prints every grid cell with its spread over training runs, and `python paper/paper_tables_recast.py
--table <name> --check` prints a Re:Cast table's numbers instead of its LaTeX.
