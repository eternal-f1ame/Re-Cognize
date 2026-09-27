# Backbones

The five backbones evaluated in the paper, and where each one comes from.

| Backbone | Source | Pinned at | Weights |
|---|---|---|---|
| TransReID | [damo-cv/TransReID](https://github.com/damo-cv/TransReID) model zoo, ViT-Base Market-1501 | SHA-256 in `weights/SHA256SUMS` | fetched by `scripts/fetch_weights.py` |
| MagiV2 | [ragavsachdeva/magiv2](https://huggingface.co/ragavsachdeva/magiv2) on the Hugging Face Hub | revision in `src/recognize/backbones.py` | loaded from the Hub |
| MagiV3 | [ragavsachdeva/magiv3](https://huggingface.co/ragavsachdeva/magiv3) on the Hugging Face Hub | revision in `src/recognize/backbones.py` | loaded from the Hub |
| InstructReID | [hwz-zju/Instruct-ReID](https://github.com/hwz-zju/Instruct-ReID) | commit `8250f44`, plus `patches/instruct-reid.patch` | fetched by `scripts/fetch_weights.py` |
| ReID5o | [Zplusdragon/ReID5o_ORBench](https://github.com/Zplusdragon/ReID5o_ORBench) | commit `6cf48e8` | downloaded by hand (below) |

## Setup

```bash
bash reid_models/setup.sh
```

The script clones Instruct-ReID into `reid_models/Instruct-ReID-main/` and applies the patch. It
copies ReID5o's code into `reid_models/ReID5o/`, together with the configuration its checkpoint
was trained with (`reid5o/configs.yaml`, from the ReID5o repository). Finally it downloads and
verifies the TransReID and InstructReID weights into `reid_models/weights/`.

The patch makes two changes. It moves one `transformers` import to `transformers.pytorch_utils`,
where the pinned `transformers` release keeps it. It also guards Instruct-ReID's joint model so
that it builds without its text branch. The code here uses only the visual encoder.

## ReID5o checkpoint

The paper uses the ReID5o checkpoint released in October 2025. Its Google Drive link (file id
`1226GUahDVeT-CyyUR8UmwOi5pu33A8-z`) appears in the ReID5o README at commit `6cf48e8`. Save it as
`reid_models/ReID5o/logs/reid5o_ckpt/best.pth` and check it:

```bash
python scripts/fetch_weights.py reid5o
```

The ReID5o authors retrained the model in January 2026 after changing its attention-head
configuration. That checkpoint is not interchangeable with the one the paper used.

## Licences

Each backbone is used under its own licence. ReID5o is released under Apache-2.0. The Magi
models and TransReID come with their own terms, and the Instruct-ReID repository states no
licence. Only this directory's setup script, patch and the ReID5o training configuration are
part of this repository. The third-party code and weights are downloaded from their sources.
