# Datasets

No images are distributed with this repository. Each corpus is obtained from its own source and
placed under `Datasets/` in one layout:

```
Datasets/<corpus>/<series>/
    images/                  page images
    annotations/             one YOLO-format .txt per page: class_id cx cy w h (normalised)
    category_mapping.json    class id -> character name
```

Pages are read in natural filename order (`page_2` before `page_10`), which is the reading order
every protocol streams in. The series-disjoint splits are pinned in `configs/training/`, and the
evaluation reads them from there.

## POPCharacters

Crop-level character identities for 23 series (chapters 1 and 2), derived from the publicly
released character annotations of PopManga ([Sachdeva et al., 2024](https://arxiv.org/abs/2401.10224)).
The split in `configs/training/data_split.yaml` has 13 training series (198 characters, 7,668
crops), 2 development series (10 characters, 873 crops) and 8 test series (70 characters, 4,058
crops).

Only the PopCharacters subset of PopManga is available online. It is the dataset
[ragavsachdeva/popmanga_test](https://huggingface.co/datasets/ragavsachdeva/popmanga_test) on the
Hugging Face Hub, gated behind its terms of use, with the code in
[ragavsachdeva/magi](https://github.com/ragavsachdeva/magi). It holds character boxes and names for
the test chapters. Its loader downloads the page images from MangaPlus, since the images stay
with their publishers.

Expected location: `Datasets/popcharacters/<series>/`, with series directory names as in the
split file (for example `Datasets/popcharacters/Dr Stone/`).

## Manga109

Request access at [manga109.org](http://www.manga109.org/) under its academic licence, then
convert the XML annotations into the layout above:

```bash
python scripts/convert_manga109.py --zip-path /path/to/Manga109_released_2023_12_07.zip \
    --split-config /tmp/manga109_split_unused.yaml
```

The converter also draws a train/val split. Send it to a scratch path as above: the evaluation
uses the committed `configs/training/data_split_manga109.yaml`, which holds out 27 volumes
(784 characters, 29,315 crops). No model is trained on Manga109. Every number reported on it is
zero-shot, from checkpoints trained on POPCharacters.

## Re:Verse

The Re:Zero benchmark of [Re:Verse](https://huggingface.co/datasets/sochastic/Re-Verse) (12
characters, 1,825 crops over 308 pages) is a gated dataset on the Hugging Face Hub, for
non-commercial research use. After accepting its terms:

```bash
git clone https://huggingface.co/datasets/sochastic/Re-Verse Datasets/Re-Verse
python scripts/prepare_reverse.py
```

The dataset ships a flat export with fifteen classes, twelve characters and three text-box
classes. `prepare_reverse.py` keeps the character boxes and writes the series directory the
evaluation reads, `Datasets/Re-Verse/Re-Zero/`, in the layout above.
