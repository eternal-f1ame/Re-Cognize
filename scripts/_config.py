"""Shared backbone registry, dataset configs, and manga lists for train.py and evaluate.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# ---------------------------------------------------------------------------
# Backbone registry
# ---------------------------------------------------------------------------

# The backbone registry lives in src/recognize/backbones.py and is re-exported here.
import sys as _sys
if str(PROJECT_ROOT / "src") not in _sys.path:
    _sys.path.insert(0, str(PROJECT_ROOT / "src"))
from recognize.backbones import ALL_BACKBONE_NAMES, BACKBONE_REGISTRY, BackboneConfig  # noqa: E402,F401

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data_dir: Path
    split_config: Optional[Path]      # None for a corpus that is evaluated whole, with no train split
    train_manga: list[str] = field(default_factory=list)
    dev_manga: list[str] = field(default_factory=list)     # checkpoint selection only
    test_manga: list[str] = field(default_factory=list)    # touched only by the final evaluation

    @property
    def val_manga(self) -> list[str]:                     # alias of test_manga, the held-out series
        return self.test_manga

    @property
    def all_manga(self) -> list[str]:
        return self.train_manga + self.dev_manga + self.test_manga


DATASET_REGISTRY: dict[str, DatasetConfig] = {
    "popcharacters": DatasetConfig(
        name="popcharacters",
        data_dir=PROJECT_ROOT / "Datasets" / "popcharacters",
        split_config=PROJECT_ROOT / "configs" / "training" / "data_split.yaml",
        train_manga=[
            "Assassination Classroom", "Black Clover", "Bleach", "Chainsaw Man",
            "Death Note", "Food Wars Shokugeki No Soma", "Hell S Paradise Jigokuraku",
            "Jujutsu Kaisen", "My Hero Academia", "Naruto", "One Piece", "Spy X Family",
            "World Trigger",
        ],
        dev_manga=["Dragon Ball", "Kuroko S Basketball"],
        test_manga=[
            "Bakuman", "Demon Slayer Kimetsu No Yaiba", "Dr Stone", "Hunter X Hunter",
            "Kagurabachi", "Nisekoi False Love", "Oshi No Ko", "Tokyo Ghoul",
        ],
    ),
    "manga109": DatasetConfig(
        name="manga109",
        data_dir=PROJECT_ROOT / "Datasets" / "manga109",
        split_config=PROJECT_ROOT / "configs" / "training" / "data_split_manga109.yaml",
        train_manga=[
            "AisazuNihaIrarenai", "AkkeraKanjinchou", "Akuhamu", "AosugiruHaru", "AppareKappore", "Arisa",
            "BEMADER_P", "Belmondo", "BokuHaSitatakaKun", "ByebyeC-BOY", "Count3DeKimeteAgeru", "DollGun",
            "Donburakokko", "DualJustice", "EienNoWith", "EvaLady", "EverydayOsakanaChan",
            "GOOD_KISS_Ver2", "Hamlet", "HaruichibanNoFukukoro", "HarukaRefrain", "HealingPlanet",
            "HeiseiJimen", "HighschoolKimengumi_vol01", "HighschoolKimengumi_vol20", "HisokaReturns",
            "JangiriPonpon", "JijiBabaFight", "Joouari", "Jyovolley", "KarappoHighschool",
            "KimiHaBokuNoTaiyouDa", "KoukouNoHitotachi", "KuroidoGanka", "LancelotFullThrottle",
            "LoveHina_vol01", "LoveHina_vol14", "MAD_STONE", "MadouTaiga", "MagicStarGakuin",
            "MariaSamaNihaNaisyo", "MayaNoAkaiKutsu", "MemorySeijin", "MeteoSanStrikeDesu",
            "MisutenaideDaisy", "MoeruOnisan_vol01", "MoeruOnisan_vol19", "MomoyamaHaikagura",
            "MukoukizuNoChonbo", "MutekiBoukenSyakuma", "Nekodama", "NichijouSoup", "Ningyoushi",
            "PLANET7", "ParaisoRoad", "PrismHeart", "Raphael", "ReveryEarth", "RinToSiteSippuNoNaka",
            "RisingGirl", "Saisoku", "SaladDays_vol01", "SaladDays_vol18", "SamayoeruSyonenNiJunaiWo",
            "SeisinkiVulnus", "ShimatteIkouze_vol01", "ShimatteIkouze_vol26", "SonokiDeABC",
            "TaiyouNiSmash", "TapkunNoTanteisitsu", "TetsuSan", "That'sIzumiko", "TouyouKidan",
            "UchiNoNyan'sDiary", "UchuKigekiM774", "UltraEleven", "UnbalanceTokyo", "YamatoNoHane",
            "YasasiiAkuma", "YouchienBoueigumi", "YoumaKourin", "YumeiroCooking",
        ],
        test_manga=[
            "ARMS", "BakuretsuKungFuGirl", "BurariTessenTorimonocho", "GakuenNoise", "GarakutayaManta",
            "GinNoChimera", "HanzaiKousyouninMinegishiEitarou", "HinagikuKenzan", "KyokugenCyclone",
            "MagicianLoad", "MiraiSan", "OL_Lunch", "OhWareraRettouSeitokai", "PikaruGenkiDesu",
            "PlatinumJungle", "PrayerHaNemurenai", "PsychoStaff", "SyabondamaKieta", "TasogareTsushin",
            "TennenSenshiG", "TensiNoHaneToAkumaNoShippo", "TotteokiNoABC", "ToutaMairimasu",
            "TsubasaNoKioku", "WarewareHaOniDearu", "YukiNoFuruMachi", "YumeNoKayoiji",
        ],
    ),
    "reverse": DatasetConfig(
        # Re:Verse, the cross-page consistency benchmark: one Re:Zero series, evaluated whole.
        # Its crops live in the canonical layout under Datasets/Re-Verse/Re-Zero (images/,
        # annotations/, category_mapping.json), which scripts/prepare_reverse.py builds from the
        # upstream flat export in Datasets/Re-Verse/data. The data is not part of the repository.
        name="reverse",
        data_dir=PROJECT_ROOT / "Datasets" / "Re-Verse",
        split_config=None,
        test_manga=["Re-Zero"],
    ),
}

ALL_DATASET_NAMES = list(DATASET_REGISTRY.keys())

# ---------------------------------------------------------------------------
# Module-level aliases for the default dataset (popcharacters)
# ---------------------------------------------------------------------------

_default_ds = DATASET_REGISTRY["popcharacters"]
TRAIN_MANGA = _default_ds.train_manga
VAL_MANGA = _default_ds.val_manga
ALL_MANGA = _default_ds.all_manga
DATA_DIR = _default_ds.data_dir
SPLIT_CONFIG = _default_ds.split_config

# ---------------------------------------------------------------------------
# Evaluation types
# ---------------------------------------------------------------------------

# The evaluation protocols are p1..p4 (scripts/evaluate.py).


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TRAIN_SCRIPT = PROJECT_ROOT / "src" / "memory_block" / "training" / "train.py"
EVALUATE_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate.py"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_dataset(name: str = "popcharacters") -> DatasetConfig:
    """Look up a dataset by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {ALL_DATASET_NAMES}")
    return DATASET_REGISTRY[name]


def dataset_prefix(dataset_name: str) -> Path:
    """Return checkpoint/results prefix for a dataset.

    popcharacters uses the repo root; other datasets get a subdirectory named after the dataset.
    """
    if dataset_name == "popcharacters":
        return PROJECT_ROOT
    return PROJECT_ROOT / dataset_name


def safe_name(manga: str) -> str:
    """Convert manga name to filesystem-safe identifier."""
    return manga.replace(" ", "_").replace("'", "")


def run_name(mode: str, backbone: str) -> str:
    """Generate the run/directory name for a combined model."""
    if mode == "baseline":
        return f"baseline_{backbone}"
    return backbone


def resolve_checkpoint_dir(mode: str, backbone: str) -> Path:
    """Return checkpoint base directory: checkpoints/{memory|baseline}/{backbone}/"""
    return PROJECT_ROOT / "checkpoints" / mode / backbone


def resolve_results_dir(mode: str, backbone: str) -> Path:
    """Return results base directory: results/{memory|baseline}/{backbone}/"""
    return PROJECT_ROOT / "results" / mode / backbone


def checkpoint_path(mode: str, backbone: str) -> Path:
    """Full path to best.pth for the combined model."""
    return resolve_checkpoint_dir(mode, backbone) / run_name(mode, backbone) / "best.pth"


def resolve_manga_list(manga_arg, split: str = "val", dataset: str = "popcharacters") -> list[str]:
    """Resolve --manga and --split flags into a concrete list for evaluation."""
    ds = resolve_dataset(dataset)
    if manga_arg == "all" or manga_arg == ["all"]:
        if split == "train":
            return ds.train_manga
        elif split == "val":
            return ds.val_manga
        return ds.all_manga
    return manga_arg if isinstance(manga_arg, list) else [manga_arg]


def resolve_backbones(backbones_arg) -> list[BackboneConfig]:
    """Resolve --backbones into a list of BackboneConfig objects."""
    if backbones_arg == "all" or backbones_arg == ["all"]:
        return list(BACKBONE_REGISTRY.values())
    return [BACKBONE_REGISTRY[b] for b in backbones_arg]
