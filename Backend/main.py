import os
import logging
import re
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 定义数据集对应的模型信息
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Data"))
ASSETS_IMAGES_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Assets", "images"))

TSNE_TIMELINE_ITERS = [0, 10_000, 50_000, 200_000]

DATASET_CONFIG = {
    "food101": {"classes": 101, "model_path": os.path.join(DATA_DIR, "food101", "model_best.pth")},
    "cifar100": {"classes": 100, "model_path": os.path.join(DATA_DIR, "cifar100", "model_best.pth")},
    "eurosat": {"classes": 10, "model_path": os.path.join(DATA_DIR, "eurosat", "model_best.pth")},
    "stl10": {"classes": 10, "model_path": os.path.join(DATA_DIR, "stl10", "model_best.pth")},
}

# 类别名称（按训练时的 label index 顺序）
# 若你的训练数据集 label 顺序不同，可以在 Data/<dataset>/labels.txt 或 labels.json 里提供自定义顺序。
DATASET_LABELS = {
    "stl10": [
        "airplane",
        "bird",
        "car",
        "cat",
        "deer",
        "dog",
        "horse",
        "monkey",
        "ship",
        "truck",
    ],
    "eurosat": [
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River",
        "SeaLake",
    ],
    # CIFAR-100 fine labels (0..99)
    "cifar100": [
        "apple",
        "aquarium_fish",
        "baby",
        "bear",
        "beaver",
        "bed",
        "bee",
        "beetle",
        "bicycle",
        "bottle",
        "bowl",
        "boy",
        "bridge",
        "bus",
        "butterfly",
        "camel",
        "can",
        "castle",
        "caterpillar",
        "cattle",
        "chair",
        "chimpanzee",
        "clock",
        "cloud",
        "cockroach",
        "couch",
        "crab",
        "crocodile",
        "cup",
        "dinosaur",
        "dolphin",
        "elephant",
        "flatfish",
        "forest",
        "fox",
        "girl",
        "hamster",
        "house",
        "kangaroo",
        "keyboard",
        "lamp",
        "lawn_mower",
        "leopard",
        "lion",
        "lizard",
        "lobster",
        "man",
        "maple_tree",
        "motorcycle",
        "mountain",
        "mouse",
        "mushroom",
        "oak_tree",
        "orange",
        "orchid",
        "otter",
        "palm_tree",
        "pear",
        "pickup_truck",
        "pine_tree",
        "plain",
        "plate",
        "poppy",
        "porcupine",
        "possum",
        "rabbit",
        "raccoon",
        "ray",
        "road",
        "rocket",
        "rose",
        "sea",
        "seal",
        "shark",
        "shrew",
        "skunk",
        "skyscraper",
        "snail",
        "snake",
        "spider",
        "squirrel",
        "streetcar",
        "sunflower",
        "sweet_pepper",
        "table",
        "tank",
        "telephone",
        "television",
        "tiger",
        "tractor",
        "train",
        "trout",
        "tulip",
        "turtle",
        "wardrobe",
        "whale",
        "willow_tree",
        "wolf",
        "woman",
        "worm",
    ],
    # Food-101 labels (0..100), standard class names
    "food101": [
        "apple_pie",
        "baby_back_ribs",
        "baklava",
        "beef_carpaccio",
        "beef_tartare",
        "beet_salad",
        "beignets",
        "bibimbap",
        "bread_pudding",
        "breakfast_burrito",
        "bruschetta",
        "caesar_salad",
        "cannoli",
        "caprese_salad",
        "carrot_cake",
        "ceviche",
        "cheesecake",
        "cheese_plate",
        "chicken_curry",
        "chicken_quesadilla",
        "chicken_wings",
        "chocolate_cake",
        "chocolate_mousse",
        "churros",
        "clam_chowder",
        "club_sandwich",
        "crab_cakes",
        "creme_brulee",
        "croque_madame",
        "cup_cakes",
        "deviled_eggs",
        "donuts",
        "dumplings",
        "edamame",
        "eggs_benedict",
        "escargots",
        "falafel",
        "filet_mignon",
        "fish_and_chips",
        "foie_gras",
        "french_fries",
        "french_onion_soup",
        "french_toast",
        "fried_calamari",
        "fried_rice",
        "frozen_yogurt",
        "garlic_bread",
        "gnocchi",
        "greek_salad",
        "grilled_cheese_sandwich",
        "grilled_salmon",
        "guacamole",
        "gyoza",
        "hamburger",
        "hot_and_sour_soup",
        "hot_dog",
        "huevos_rancheros",
        "hummus",
        "ice_cream",
        "lasagna",
        "lobster_bisque",
        "lobster_roll_sandwich",
        "macaroni_and_cheese",
        "macarons",
        "miso_soup",
        "mussels",
        "nachos",
        "omelette",
        "onion_rings",
        "oysters",
        "pad_thai",
        "paella",
        "pancakes",
        "panna_cotta",
        "peking_duck",
        "pho",
        "pizza",
        "pork_chop",
        "poutine",
        "prime_rib",
        "pulled_pork_sandwich",
        "ramen",
        "ravioli",
        "red_velvet_cake",
        "risotto",
        "samosa",
        "sashimi",
        "scallops",
        "seaweed_salad",
        "shrimp_and_grits",
        "spaghetti_bolognese",
        "spaghetti_carbonara",
        "spring_rolls",
        "steak",
        "strawberry_shortcake",
        "sushi",
        "tacos",
        "takoyaki",
        "tiramisu",
        "tuna_tartare",
        "waffles",
    ],
}


def _load_labels_from_data_dir(dataset: str):
    import json

    dataset_dir = os.path.join(DATA_DIR, dataset)
    json_path = os.path.join(dataset_dir, "labels.json")
    txt_path = os.path.join(dataset_dir, "labels.txt")

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data

    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
        lines = [x for x in lines if x]
        if lines:
            return lines

    return None


def get_label_name(dataset: str, class_index: int) -> str:
    labels = _load_labels_from_data_dir(dataset) or DATASET_LABELS.get(dataset)
    if isinstance(labels, list) and 0 <= class_index < len(labels):
        return labels[class_index]
    return f"类别 {class_index}"


_DATASET_LABELS_ZH = {
    "stl10": {
        "airplane": "飞机",
        "bird": "鸟",
        "car": "汽车",
        "cat": "猫",
        "deer": "鹿",
        "dog": "狗",
        "horse": "马",
        "monkey": "猴",
        "ship": "船",
        "truck": "卡车",
    },
    "eurosat": {
        "AnnualCrop": "年生作物",
        "Forest": "森林",
        "HerbaceousVegetation": "草本植被",
        "Highway": "公路",
        "Industrial": "工业区",
        "Pasture": "牧场",
        "PermanentCrop": "多年生作物",
        "Residential": "居民区",
        "River": "河流",
        "SeaLake": "海/湖",
    },
}


_ZH_PHRASE_MAP = {
    # CIFAR-100 common multi-token items
    "aquarium_fish": "观赏鱼",
    "lawn_mower": "割草机",
    "maple_tree": "枫树",
    "oak_tree": "橡树",
    "palm_tree": "棕榈树",
    "pine_tree": "松树",
    "pickup_truck": "皮卡车",
    "sweet_pepper": "甜椒",
    "streetcar": "有轨电车",
    "flatfish": "比目鱼",
    # Food-101 common multi-token items
    "hot_dog": "热狗",
    "french_fries": "薯条",
    "french_toast": "法式吐司",
    "fried_rice": "炒饭",
    "garlic_bread": "蒜蓉面包",
    "ice_cream": "冰淇淋",
    "macaroni_and_cheese": "芝士通心粉",
    "pulled_pork_sandwich": "手撕猪肉三明治",
    "shrimp_and_grits": "虾配玉米粥",
    "spaghetti_bolognese": "肉酱意面",
    "spaghetti_carbonara": "培根蛋奶意面",
    "strawberry_shortcake": "草莓奶油蛋糕",
    "tuna_tartare": "金枪鱼塔塔",
    "pancakes": "煎饼",
}


_ZH_WORD_MAP = {
    # general
    "apple": "苹果",
    "orange": "橙子",
    "pear": "梨",
    "peach": "桃",
    "grape": "葡萄",
    "banana": "香蕉",
    "lemon": "柠檬",
    "mushroom": "蘑菇",
    "forest": "森林",
    "mountain": "山",
    "road": "道路",
    "bridge": "桥",
    "house": "房子",
    "skyscraper": "摩天楼",
    "cloud": "云",
    "sea": "海",
    "river": "河",
    "plain": "平原",
    # animals
    "bear": "熊",
    "beaver": "海狸",
    "bee": "蜜蜂",
    "beetle": "甲虫",
    "butterfly": "蝴蝶",
    "camel": "骆驼",
    "caterpillar": "毛毛虫",
    "cattle": "牛",
    "chimpanzee": "黑猩猩",
    "cockroach": "蟑螂",
    "crab": "螃蟹",
    "crocodile": "鳄鱼",
    "dinosaur": "恐龙",
    "dolphin": "海豚",
    "elephant": "大象",
    "fox": "狐狸",
    "hamster": "仓鼠",
    "kangaroo": "袋鼠",
    "leopard": "豹",
    "lion": "狮子",
    "lizard": "蜥蜴",
    "lobster": "龙虾",
    "mouse": "老鼠",
    "otter": "水獭",
    "porcupine": "豪猪",
    "possum": "负鼠",
    "rabbit": "兔子",
    "raccoon": "浣熊",
    "ray": "鳐鱼",
    "seal": "海豹",
    "shark": "鲨鱼",
    "shrew": "鼩鼱",
    "skunk": "臭鼬",
    "snail": "蜗牛",
    "snake": "蛇",
    "spider": "蜘蛛",
    "squirrel": "松鼠",
    "tiger": "老虎",
    "trout": "鳟鱼",
    "turtle": "乌龟",
    "whale": "鲸",
    "wolf": "狼",
    "worm": "虫",
    # vehicles & objects
    "bicycle": "自行车",
    "bus": "公交车",
    "motorcycle": "摩托车",
    "rocket": "火箭",
    "tank": "坦克",
    "tractor": "拖拉机",
    "train": "火车",
    "airplane": "飞机",
    "ship": "船",
    "truck": "卡车",
    "car": "汽车",
    "chair": "椅子",
    "couch": "沙发",
    "table": "桌子",
    "wardrobe": "衣柜",
    "bed": "床",
    "bottle": "瓶子",
    "bowl": "碗",
    "can": "罐头",
    "cup": "杯子",
    "plate": "盘子",
    "clock": "时钟",
    "keyboard": "键盘",
    "lamp": "灯",
    "telephone": "电话",
    "television": "电视",
    # food words (for heuristic Food101)
    "pie": "派",
    "ribs": "肋排",
    "salad": "沙拉",
    "beef": "牛肉",
    "chicken": "鸡肉",
    "pork": "猪肉",
    "fish": "鱼",
    "shrimp": "虾",
    "soup": "汤",
    "cake": "蛋糕",
    "chocolate": "巧克力",
    "cheese": "奶酪",
    "pizza": "披萨",
    "sushi": "寿司",
    "tacos": "塔可",
    "tuna": "金枪鱼",
    "tartare": "塔塔",
    "ramen": "拉面",
    "dumplings": "饺子",
    "donuts": "甜甜圈",
    "pancakes": "煎饼",
    "waffles": "华夫饼",
}


def _split_label_tokens(label: str) -> List[str]:
    if "_" in label:
        return [t for t in label.split("_") if t]
    # Split CamelCase (EuroSAT) into tokens
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", label)
    return [p for p in parts if p]


def get_label_name_zh(dataset: str, label_en: str) -> str:
    dataset_map = _DATASET_LABELS_ZH.get(dataset)
    if isinstance(dataset_map, dict) and label_en in dataset_map:
        return dataset_map[label_en]

    # Phrase-level mapping (best effort)
    if label_en in _ZH_PHRASE_MAP:
        return _ZH_PHRASE_MAP[label_en]

    tokens = _split_label_tokens(label_en)
    translated = []
    for token in tokens:
        key = token.lower()
        zh = _ZH_WORD_MAP.get(key)
        if zh is None and key.endswith("s"):
            zh = _ZH_WORD_MAP.get(key[:-1])
        translated.append(zh or token)

    # Prefer compact join when everything looks CJK; otherwise keep readable spaces
    if translated and all(re.search(r"[\u4e00-\u9fff]", x) for x in translated):
        return "".join(translated)
    return " ".join(translated) if translated else label_en

# 运行时缓存已加载的模型，避免重复加载
MODELS = {}
MODEL_INPUT_SIZES = {}

# 可视化/报告数据缓存（只读）
VIZ_CACHE = {
    "coords": {},
    "curves": {},
    "pseudo_conf": {},
    "cm": {},
    "features": {},
    "feature_labels": {},
    "centroids": {},
    "text_match": {},
    "asset_index": {},
}


# 静态资源：示例图片（Assets/images/...）
if os.path.isdir(ASSETS_IMAGES_DIR):
    app.mount("/assets/images", StaticFiles(directory=ASSETS_IMAGES_DIR), name="assets-images")


def _ensure_dataset_name(dataset: str) -> str:
    if dataset not in DATASET_CONFIG:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {dataset}")
    return dataset


def _read_json(path: str):
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_data_paths(dataset: str):
    dataset_dir = os.path.join(DATA_DIR, dataset)
    return {
        # New naming: explicit initial/final
        "coords_initial": os.path.join(dataset_dir, f"coords_{dataset}_initial.json"),
        "coords_final": os.path.join(dataset_dir, f"coords_{dataset}_final.json"),
        # Backward-compatible naming
        "coords": os.path.join(dataset_dir, f"coords_{dataset}.json"),
        "coords_tpl": os.path.join(dataset_dir, f"coords_{dataset}_{{iter}}.json"),
        "curves": os.path.join(dataset_dir, f"curves_{dataset}.json"),
        "pseudo_conf_npy": os.path.join(dataset_dir, f"pseudo_label_confidences_{dataset}.npy"),
        "cm": os.path.join(dataset_dir, f"cm_{dataset}.json"),
        "features_npy": os.path.join(dataset_dir, f"features_{dataset}.npy"),
        "features_meta": os.path.join(dataset_dir, f"features_{dataset}.json"),
        "text_match_npy": os.path.join(dataset_dir, f"text_match_{dataset}.npy"),
        "text_match_json": os.path.join(dataset_dir, f"text_match_{dataset}.json"),
    }


def _load_coords(dataset: str, iteration: Optional[int] = None):
    cache: Dict[Tuple[str, Optional[int]], list] = VIZ_CACHE["coords"]
    key = (dataset, iteration)
    if key in cache:
        return cache[key]

    paths = _get_data_paths(dataset)
    if iteration is None:
        # Prefer explicit final file; fallback to legacy coords_<dataset>.json
        path = paths["coords_final"] if os.path.exists(paths["coords_final"]) else paths["coords"]
    else:
        it = int(iteration)
        # For Initial stage, prefer explicit initial file when iteration==0
        if it == 0 and os.path.exists(paths["coords_initial"]):
            path = paths["coords_initial"]
        else:
            path = paths["coords_tpl"].format(iter=it)

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="coords file not found")

    data = _read_json(path)
    if not isinstance(data, list):
        raise HTTPException(status_code=500, detail="coords json must be a list")

    cache[key] = data
    return data


def _timeline_progress(iteration: int) -> float:
    # Map to [0,1] based on TSNE_TIMELINE_ITERS.
    if iteration <= TSNE_TIMELINE_ITERS[0]:
        return 0.0
    if iteration >= TSNE_TIMELINE_ITERS[-1]:
        return 1.0
    for a, b in zip(TSNE_TIMELINE_ITERS, TSNE_TIMELINE_ITERS[1:]):
        if a <= iteration <= b:
            return (iteration - a) / float(b - a)
    return 1.0


def _generate_synthetic_coords(dataset: str, iteration: int) -> list:
    """当缺少指定 iteration 的 coords 文件时：用最终 coords 生成一个“从乱到聚集”的可视化降级版本。

    说明：这不是严格意义上的真实训练时刻投影，只用于保证前端时间轴交互可工作。
    """

    final_points = _load_coords(dataset, iteration=None)
    n = len(final_points)
    if n == 0:
        return []

    xs = np.fromiter((float(p.get("x", 0.0)) for p in final_points if isinstance(p, dict)), dtype=np.float32, count=n)
    ys = np.fromiter((float(p.get("y", 0.0)) for p in final_points if isinstance(p, dict)), dtype=np.float32, count=n)
    labels = np.fromiter((int(p.get("label", -1)) for p in final_points if isinstance(p, dict)), dtype=np.int32, count=n)

    prog = _timeline_progress(int(iteration))
    # 越早越“乱”：随机底图权重更高
    seed = (abs(hash(dataset)) % (2**31 - 1))
    rng = np.random.default_rng(seed)

    # 用最终坐标的方差来设置随机底图尺度
    sx = float(np.std(xs)) or 1.0
    sy = float(np.std(ys)) or 1.0
    rand_x = rng.normal(0.0, sx, size=n).astype(np.float32)
    rand_y = rng.normal(0.0, sy, size=n).astype(np.float32)

    # 线性插值：t=0 -> rand, t=1 -> final
    t = np.float32(prog)
    out_x = (1.0 - t) * rand_x + t * xs
    out_y = (1.0 - t) * rand_y + t * ys

    data = []
    for i in range(n):
        data.append({"x": float(out_x[i]), "y": float(out_y[i]), "label": int(labels[i])})
    return data


def _load_curves(dataset: str):
    cache = VIZ_CACHE["curves"]
    if dataset in cache:
        return cache[dataset]

    paths = _get_data_paths(dataset)
    if not os.path.exists(paths["curves"]):
        raise HTTPException(status_code=404, detail="curves file not found")

    data = _read_json(paths["curves"])
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="curves json must be an object")

    cache[dataset] = data
    return data


def _load_pseudo_confidences(dataset: str):
    cache = VIZ_CACHE["pseudo_conf"]
    if dataset in cache:
        return cache[dataset]

    paths = _get_data_paths(dataset)
    if os.path.exists(paths["pseudo_conf_npy"]):
        arr = np.load(paths["pseudo_conf_npy"])
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        cache[dataset] = arr
        return arr

    raise HTTPException(status_code=404, detail="pseudo confidence file not found")


def _load_confusion_matrix(dataset: str):
    cache = VIZ_CACHE["cm"]
    if dataset in cache:
        return cache[dataset]

    paths = _get_data_paths(dataset)
    if not os.path.exists(paths["cm"]):
        raise HTTPException(status_code=404, detail="confusion matrix file not found")

    data = _read_json(paths["cm"])
    if not isinstance(data, dict) or "confusion_matrix" not in data:
        raise HTTPException(status_code=500, detail="cm json missing confusion_matrix")

    cache[dataset] = data
    return data


def _preprocess_pil_image(image: Image.Image, size: int = 224) -> torch.Tensor:
    # Resize
    image = image.convert("RGB").resize((size, size))

    # ToTensor (C, H, W) in [0,1]
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected an RGB image")
    tensor = torch.from_numpy(arr).permute(2, 0, 1)

    # Normalize (ImageNet)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype)[:, None, None]
    tensor = (tensor - mean) / std

    return tensor.unsqueeze(0)


def _load_features(dataset: str) -> np.ndarray:
    cache = VIZ_CACHE["features"]
    if dataset in cache:
        return cache[dataset]

    paths = _get_data_paths(dataset)
    if not os.path.exists(paths["features_npy"]):
        raise HTTPException(status_code=404, detail="features file not found")

    arr = np.load(paths["features_npy"], mmap_mode="r")
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise HTTPException(status_code=500, detail="features npy must be 2D")

    cache[dataset] = arr
    return arr


def _load_feature_labels(dataset: str) -> np.ndarray:
    cache = VIZ_CACHE["feature_labels"]
    if dataset in cache:
        return cache[dataset]

    paths = _get_data_paths(dataset)
    labels = None

    if os.path.exists(paths["features_meta"]):
        meta = _read_json(paths["features_meta"])
        if isinstance(meta, dict) and isinstance(meta.get("labels"), list):
            labels = meta.get("labels")

    if labels is None:
        coords = _load_coords(dataset, iteration=None)
        labels = [p.get("label", -1) for p in coords if isinstance(p, dict)]

    arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    cache[dataset] = arr
    return arr


def _get_normalized_centroids(dataset: str) -> np.ndarray:
    cache = VIZ_CACHE["centroids"]
    if dataset in cache:
        return cache[dataset]

    features = _load_features(dataset).astype(np.float32, copy=False)
    labels = _load_feature_labels(dataset)
    if features.shape[0] != labels.shape[0]:
        raise HTTPException(status_code=500, detail="features and labels length mismatch")

    num_classes = int(DATASET_CONFIG[dataset]["classes"])
    dim = int(features.shape[1])

    sums = np.zeros((num_classes, dim), dtype=np.float64)
    counts = np.zeros((num_classes,), dtype=np.int64)

    valid = (labels >= 0) & (labels < num_classes)
    lbl = labels[valid]
    feat = features[valid]
    np.add.at(sums, lbl, feat)
    np.add.at(counts, lbl, 1)

    counts_safe = np.maximum(counts, 1)[:, None]
    centroids = (sums / counts_safe).astype(np.float32)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    centroids = centroids / norms

    cache[dataset] = centroids
    return centroids


def _cosine_topk_to_labels(dataset: str, scores: np.ndarray, k: int) -> list:
    k = int(max(1, min(int(k), int(scores.size))))
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]

    out = []
    for class_index in idx.tolist():
        class_index = int(class_index)
        label_en = get_label_name(dataset, class_index)
        label_zh = get_label_name_zh(dataset, label_en)
        score = float(scores[class_index])
        out.append(
            {
                "index": class_index,
                "label_en": label_en,
                "label_zh": label_zh,
                "label": f"{label_en}（{label_zh}）" if label_zh else label_en,
                "score": score,
                "score_01": float((score + 1.0) / 2.0),
            }
        )
    return out


def _detect_score_type(scores: np.ndarray) -> str:
    """Best-effort detection of score semantics.

    - If values are mostly within [0,1] and sum is ~1 => "prob" (softmax probabilities)
    - Otherwise => "score" (e.g. cosine similarity or logits)
    """

    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return "score"
    s = float(np.sum(scores))
    mn = float(np.min(scores))
    mx = float(np.max(scores))
    if mn >= -1e-6 and mx <= 1.0 + 1e-6 and abs(s - 1.0) <= 1e-2:
        return "prob"
    return "score"


def _topk_to_labels(dataset: str, scores: np.ndarray, k: int, score_type: str) -> list:
    """Convert a score vector into Top-K label rows.

    score_type:
      - "prob": scores are probabilities in [0,1], sum~1
      - "score": arbitrary scores (e.g. cosine), possibly in [-1,1]
    """

    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    k = int(max(1, min(int(k), int(scores.size))))
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]

    out = []
    for class_index in idx.tolist():
        class_index = int(class_index)
        label_en = get_label_name(dataset, class_index)
        label_zh = get_label_name_zh(dataset, label_en)
        score = float(scores[class_index])

        if score_type == "prob":
            score_01 = score
        else:
            score_01 = float((score + 1.0) / 2.0)

        out.append(
            {
                "index": class_index,
                "label_en": label_en,
                "label_zh": label_zh,
                "label": f"{label_en}（{label_zh}）" if label_zh else label_en,
                "score": score,
                "score_01": score_01,
            }
        )
    return out


def _load_text_match_matrix(dataset: str) -> np.ndarray:
    """Load and validate Data/<dataset>/text_match_<dataset>.npy.

    Ensures:
    - 2D shape
    - columns == num_classes
    - rows == number of coords points (final)
    """

    paths = _get_data_paths(dataset)
    if not os.path.exists(paths["text_match_npy"]):
        raise HTTPException(status_code=404, detail="text_match npy not found")

    tm_cache = VIZ_CACHE["text_match"]
    if dataset in tm_cache:
        tm = np.asarray(tm_cache[dataset])
    else:
        tm = np.load(paths["text_match_npy"], mmap_mode="r")
        tm = np.asarray(tm)
        tm_cache[dataset] = tm

    if tm.ndim != 2:
        raise HTTPException(status_code=500, detail="text_match npy must be 2D")

    expected_cols = int(DATASET_CONFIG[dataset]["classes"])
    if int(tm.shape[1]) != expected_cols:
        raise HTTPException(
            status_code=500,
            detail=f"text_match columns mismatch: got {tm.shape[1]}, expected {expected_cols}",
        )

    # Align rows with coords length (final)
    try:
        expected_rows = int(len(_load_coords(dataset, iteration=None)))
    except Exception:
        expected_rows = None

    if expected_rows is not None and int(tm.shape[0]) != expected_rows:
        raise HTTPException(
            status_code=500,
            detail=f"text_match rows mismatch: got {tm.shape[0]}, expected {expected_rows}",
        )

    return tm


_ASSET_RE = re.compile(r"^idx_(?P<idx>\d+)_target_(?P<target>\d+)_pred_(?P<pred>\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def _build_asset_index(dataset: str) -> Dict[int, dict]:
    cache = VIZ_CACHE["asset_index"]
    if dataset in cache:
        return cache[dataset]

    index: Dict[int, dict] = {}
    base = os.path.join(ASSETS_IMAGES_DIR, dataset)
    for group in ["correct", "errors"]:
        folder = os.path.join(base, group)
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            m = _ASSET_RE.match(name)
            if not m:
                continue
            idx = int(m.group("idx"))
            if idx in index:
                continue
            target = int(m.group("target"))
            pred = int(m.group("pred"))
            url = f"/assets/images/{dataset}/{group}/{name}"
            index[idx] = {"idx": idx, "group": group, "target": target, "pred": pred, "url": url}

    cache[dataset] = index
    return index


def _parse_vit_arch(arch: str) -> dict:
    """Parse strings like 'vit_base_patch16_96' into components."""
    parts = arch.split("_")
    if len(parts) != 4 or parts[0] != "vit":
        raise ValueError(f"Unsupported arch format: {arch}")

    variant = parts[1]
    patch_part = parts[2]
    img_part = parts[3]
    if not patch_part.startswith("patch"):
        raise ValueError(f"Unsupported patch spec in arch: {arch}")

    patch_size = int(patch_part[len("patch"):])
    img_size = int(img_part)
    return {"variant": variant, "patch_size": patch_size, "img_size": img_size}


def _create_vit_model(arch: str, num_classes: int):
    """Create ViT model from timm name, with fallback for custom sizes."""
    import timm

    # timm 已注册的模型名直接创建
    if arch in timm.list_models(pretrained=False):
        return timm.create_model(arch, pretrained=False, num_classes=num_classes)

    # 自定义（例如 vit_base_patch16_96 / vit_small_patch2_32）
    cfg = _parse_vit_arch(arch)
    presets = {
        "base": {"embed_dim": 768, "depth": 12, "num_heads": 12, "mlp_ratio": 4.0},
        "small": {"embed_dim": 384, "depth": 12, "num_heads": 6, "mlp_ratio": 4.0},
        "tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3, "mlp_ratio": 4.0},
    }
    if cfg["variant"] not in presets:
        raise ValueError(f"Unsupported ViT variant '{cfg['variant']}' for arch '{arch}'")

    from timm.models.vision_transformer import VisionTransformer

    preset = presets[cfg["variant"]]
    return VisionTransformer(
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        in_chans=3,
        num_classes=num_classes,
        embed_dim=preset["embed_dim"],
        depth=preset["depth"],
        num_heads=preset["num_heads"],
        mlp_ratio=preset["mlp_ratio"],
        qkv_bias=True,
    )


def _looks_like_state_dict(obj) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    # state_dict 通常是: {str: Tensor}
    sample_value = next(iter(obj.values()))
    return isinstance(sample_value, torch.Tensor)


def _extract_state_dict(checkpoint: object) -> object:
    """从各种训练 checkpoint 结构中提取可用于 load_state_dict 的权重字典。"""
    if isinstance(checkpoint, dict):
        # 常见保存结构：{'state_dict': ...} 或 {'model': ...} / {'ema_model': ...}
        # 注意：有些工程会在 checkpoint['net'] 里存模型名字符串（例如 vit_base_patch16_224），
        # 这不是权重本体，所以不要在这里把 'net' 当作 state_dict 提取。
        for key in ("ema_model", "model", "state_dict", "model_state_dict", "network", "student"):
            if key in checkpoint:
                candidate = checkpoint[key]
                if isinstance(candidate, dict) and "state_dict" in candidate and isinstance(candidate["state_dict"], dict):
                    return candidate["state_dict"]
                return candidate

        # 有些训练代码直接把 state_dict 存在顶层
        if _looks_like_state_dict(checkpoint):
            return checkpoint

    return checkpoint


def _infer_vit_from_state_dict(state_dict: dict, num_classes: int):
    """Infer ViT config from state_dict tensor shapes and build VisionTransformer.

    Works for timm-style keys (cls_token/pos_embed/patch_embed.proj/blocks.*).
    """
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("state_dict must be a non-empty dict")

    # Some checkpoints save weights with a fixed prefix (e.g. DataParallel adds 'module.').
    # Strip common prefixes first so we can find timm-style keys.
    for prefix in ("module.", "model.", "backbone.", "encoder.", "net."):
        if isinstance(state_dict, dict) and state_dict and all(isinstance(k, str) and k.startswith(prefix) for k in state_dict.keys()):
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}

    if "patch_embed.proj.weight" not in state_dict:
        # Fallback: find any key ending with patch_embed.proj.weight and strip that prefix
        suffix = "patch_embed.proj.weight"
        match = next((k for k in state_dict.keys() if isinstance(k, str) and k.endswith(suffix)), None)
        if match is not None and match != suffix:
            inferred_prefix = match[: -len(suffix)]
            state_dict = {k[len(inferred_prefix):]: v for k, v in state_dict.items() if isinstance(k, str) and k.startswith(inferred_prefix)}

    if "patch_embed.proj.weight" not in state_dict:
        raise ValueError("state_dict missing patch_embed.proj.weight; cannot infer ViT")

    patch_w = state_dict["patch_embed.proj.weight"]
    if not isinstance(patch_w, torch.Tensor) or patch_w.ndim != 4:
        raise ValueError("patch_embed.proj.weight has unexpected shape")

    embed_dim = int(patch_w.shape[0])
    patch_size = int(patch_w.shape[2])

    # Infer depth from blocks.*
    block_ids = set()
    for k in state_dict.keys():
        if isinstance(k, str) and k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_ids.add(int(parts[1]))
    depth = (max(block_ids) + 1) if block_ids else 12

    # Infer num_heads from embed_dim (common timm convention uses 64 dim per head)
    num_heads = max(1, embed_dim // 64)

    # Infer img_size from pos_embed length if available
    img_size = 224
    pos = state_dict.get("pos_embed")
    if isinstance(pos, torch.Tensor) and pos.ndim == 3 and pos.shape[1] > 1:
        num_tokens = int(pos.shape[1])
        num_patches = num_tokens - 1  # minus cls token
        grid = int(round(num_patches ** 0.5))
        if grid * grid == num_patches:
            img_size = grid * patch_size

    from timm.models.vision_transformer import VisionTransformer

    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
    ), img_size


def _strip_prefix_if_present(state_dict: dict, prefix: str) -> dict:
    if not state_dict:
        return state_dict
    if all(isinstance(k, str) and k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _maybe_strip_known_prefixes(state_dict: dict, model: torch.nn.Module) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict

    model_keys = set(model.state_dict().keys())
    if not model_keys:
        return state_dict

    # 先处理最常见的 DataParallel 前缀
    state_dict = _strip_prefix_if_present(state_dict, "module.")

    # 再尝试若权重被包了一层命名前缀（例如 'model.'/'backbone.'）
    prefixes = ("model.", "backbone.", "encoder.", "net.")
    for prefix in prefixes:
        stripped = _strip_prefix_if_present(state_dict, prefix)
        if stripped is not state_dict and any(k in model_keys for k in stripped.keys()):
            return stripped

    return state_dict

def get_model(dataset_name):
    if dataset_name in MODELS:
        return MODELS[dataset_name]
    
    config = DATASET_CONFIG.get(dataset_name)
    if not config or not os.path.exists(config["model_path"]):
        return None
    
    try:
        # 先加载 checkpoint（CPU）
        try:
            checkpoint = torch.load(config["model_path"], map_location="cpu", weights_only=False)
        except TypeError:
            # 兼容旧版本 torch（没有 weights_only 参数）
            checkpoint = torch.load(config["model_path"], map_location="cpu")

        # 从 checkpoint 推断网络结构（不同数据集模型可能不同）
        arch = None
        if isinstance(checkpoint, dict):
            for k in ("net", "arch", "model_name"):
                if isinstance(checkpoint.get(k), str) and checkpoint.get(k).strip():
                    arch = checkpoint.get(k).strip()
                    break

        state_dict = _extract_state_dict(checkpoint)
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError("Checkpoint does not contain a valid state_dict")

        # 若 checkpoint 没带 arch/net 字符串，直接从权重形状推断（支持 vit_base_patch16_96 / vit_small_patch2_32 等）
        inferred_img_size = None
        if arch is None:
            model, inferred_img_size = _infer_vit_from_state_dict(state_dict, num_classes=config["classes"])
        else:
            try:
                model = _create_vit_model(arch, num_classes=config["classes"])
            except Exception:
                # 如果 arch 有但创建/形状不匹配，回退到从权重推断
                logger.warning("Arch '%s' failed for %s; falling back to infer-from-state_dict", arch, dataset_name)
                model, inferred_img_size = _infer_vit_from_state_dict(state_dict, num_classes=config["classes"])

        state_dict = _maybe_strip_known_prefixes(state_dict, model)

        # 训练 checkpoint 可能包含额外键（如 project_layer.*），用 strict=False 更稳妥
        incompatible = model.load_state_dict(state_dict, strict=False)
        if getattr(incompatible, "missing_keys", None) or getattr(incompatible, "unexpected_keys", None):
            if getattr(incompatible, "missing_keys", None):
                logger.warning(
                    "Missing keys when loading %s (%s): %s",
                    dataset_name,
                    arch,
                    incompatible.missing_keys[:30],
                )
            if getattr(incompatible, "unexpected_keys", None):
                logger.info(
                    "Unexpected keys when loading %s (%s): %s",
                    dataset_name,
                    arch,
                    incompatible.unexpected_keys[:30],
                )

        model.eval()

        # 缓存该数据集的输入尺寸，用于推理时 resize
        if inferred_img_size is not None:
            MODEL_INPUT_SIZES[dataset_name] = inferred_img_size
        else:
            try:
                MODEL_INPUT_SIZES[dataset_name] = _parse_vit_arch(arch)["img_size"]
            except Exception:
                MODEL_INPUT_SIZES[dataset_name] = 224
    except Exception:
        logger.exception("Failed to load model for dataset=%s from %s", dataset_name, config.get("model_path"))
        return None
    
    MODELS[dataset_name] = model
    return model

# 2. 预测接口
@app.post("/api/predict/{dataset}")
async def predict(dataset: str, file: UploadFile = File(...)):
    model = get_model(dataset)
    if not model:
        return {"success": False, "error": f"Model for {dataset} not found"}

    # 图像预处理（不依赖 torchvision）
    image = Image.open(file.file)
    img_size = MODEL_INPUT_SIZES.get(dataset, 224)
    img_tensor = _preprocess_pil_image(image, size=img_size)

    # 推理
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_idx = torch.topk(prob, 5)

    predictions = []
    for p, idx in zip(top5_prob, top5_idx):
        class_index = int(idx.item())
        label_en = get_label_name(dataset, class_index)
        label_zh = get_label_name_zh(dataset, label_en)
        predictions.append(
            {
                "index": class_index,
                "label_en": label_en,
                "label_zh": label_zh,
                "label": f"{label_en}（{label_zh}）" if label_zh else label_en,
                "score": float(p),
            }
        )

    return {"success": True, "predictions": predictions}


@app.get("/api/meta/datasets")
def list_datasets():
    domain_map = {
        "food101": "美食",
        "stl10": "工业",
        "eurosat": "遥感",
        "cifar100": "通用",
    }

    rows = []
    for name, cfg in DATASET_CONFIG.items():
        paths = _get_data_paths(name)

        # coords: support explicit initial/final + legacy single file + per-iter timeline files
        has_coords_any = (
            os.path.exists(paths["coords_initial"]) or
            os.path.exists(paths["coords_final"]) or
            os.path.exists(paths["coords"]) or
            any(
            os.path.exists(paths["coords_tpl"].format(iter=int(it))) for it in TSNE_TIMELINE_ITERS
            )
        )
        rows.append(
            {
                "dataset": name,
                "domain": domain_map.get(name, ""),
                "num_classes": int(cfg["classes"]),
                "has_coords": has_coords_any,
                "has_curves": os.path.exists(paths["curves"]),
                "has_pseudo_conf": os.path.exists(paths["pseudo_conf_npy"]),
                "has_cm": os.path.exists(paths["cm"]),
            }
        )
    return {"rows": rows}


@app.get("/api/viz/{dataset}/coords")
def get_tsne_coords(
    dataset: str,
    iteration: Optional[int] = Query(None, ge=0),
    limit: int = Query(5000, ge=100, le=50000),
):
    dataset = _ensure_dataset_name(dataset)

    source = "file"
    missing = False
    used_iteration = None if iteration is None else int(iteration)

    # 读取指定 iteration 的文件；缺失则用合成降级数据
    if used_iteration is None:
        data = _load_coords(dataset, iteration=None)
        used_iteration = TSNE_TIMELINE_ITERS[-1]
    else:
        try:
            data = _load_coords(dataset, iteration=used_iteration)
        except HTTPException:
            data = _generate_synthetic_coords(dataset, used_iteration)
            source = "synthetic"
            missing = True

    total = len(data)

    points = data
    if total > limit:
        rng = np.random.default_rng(0)
        idx = rng.choice(total, size=limit, replace=False)
        idx.sort()
        points = [data[int(i)] for i in idx]

    asset_index = _build_asset_index(dataset)

    # 保证类型可 JSON 序列化
    cleaned = []
    for idx, p in enumerate(points):
        if not isinstance(p, dict):
            continue
        sample_idx = int(p.get("idx", idx))
        asset = asset_index.get(sample_idx)
        cleaned.append(
            {
                "x": float(p.get("x", 0.0)),
                "y": float(p.get("y", 0.0)),
                "label": int(p.get("label", -1)),
                "idx": sample_idx,
                "image_url": (asset.get("url") if asset else None),
                "target": (asset.get("target") if asset else None),
                "pred": (asset.get("pred") if asset else None),
                "group": (asset.get("group") if asset else None),
            }
        )

    return {
        "dataset": dataset,
        "iteration": int(used_iteration),
        "timeline_iters": TSNE_TIMELINE_ITERS,
        "source": source,
        "missing": bool(missing),
        "total": total,
        "returned": len(cleaned),
        "points": cleaned,
    }


@app.get("/api/viz/{dataset}/curves")
def get_curves(dataset: str):
    dataset = _ensure_dataset_name(dataset)
    return _load_curves(dataset)


@app.get("/api/viz/{dataset}/pseudo-confidence")
def get_pseudo_confidence(
    dataset: str,
    threshold: float = Query(0.95, ge=0.0, le=1.0),
    bins: int = Query(20, ge=5, le=100),
):
    dataset = _ensure_dataset_name(dataset)
    conf = _load_pseudo_confidences(dataset)
    conf = np.asarray(conf, dtype=np.float32)
    if conf.size == 0:
        raise HTTPException(status_code=500, detail="empty pseudo confidence array")

    bin_edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float32)
    bin_counts, _ = np.histogram(conf, bins=bin_edges)
    pass_ratio = float((conf >= threshold).mean())
    passed = conf[conf >= threshold]
    pass_mean = (None if passed.size == 0 else float(passed.mean()))

    return {
        "dataset": dataset,
        "n": int(conf.size),
        "min": float(conf.min()),
        "max": float(conf.max()),
        "mean": float(conf.mean()),
        "threshold": float(threshold),
        "pass_ratio": pass_ratio,
        "pass_mean_confidence": pass_mean,
        "bin_edges": [float(x) for x in bin_edges.tolist()],
        "bin_counts": [int(x) for x in bin_counts.tolist()],
    }


@app.get("/api/viz/{dataset}/samples")
def list_sample_images(
    dataset: str,
    limit: int = Query(200, ge=1, le=5000),
):
    dataset = _ensure_dataset_name(dataset)
    index = _build_asset_index(dataset)
    items = list(index.values())
    items.sort(key=lambda x: x["idx"])
    if len(items) > limit:
        items = items[: int(limit)]
    return {"dataset": dataset, "total": int(len(index)), "returned": int(len(items)), "samples": items}


@app.get("/api/viz/{dataset}/text-match/topk")
def get_text_match_topk(
    dataset: str,
    idx: int = Query(..., ge=0),
    k: int = Query(5, ge=1, le=20),
):
    """CLIP 文本-视觉匹配 Top-K。

    优先读取 Data/<dataset>/text_match_<dataset>.npy/json（如果你已有 text_match.py 的预计算产物）。
    若缺失，则用 features 向量 + 类别中心点做一个可用的 cosine-sim 降级版本。
    """

    dataset = _ensure_dataset_name(dataset)
    paths = _get_data_paths(dataset)
    mode = "fallback_feature_centroid"

    asset = _build_asset_index(dataset).get(int(idx))
    image_url = (asset.get("url") if asset else None)

    # 1) 优先：预计算矩阵（npy）
    if os.path.exists(paths["text_match_npy"]):
        tm = _load_text_match_matrix(dataset)
        if not (0 <= int(idx) < int(tm.shape[0])):
            raise HTTPException(status_code=404, detail="idx out of range")

        scores = np.asarray(tm[int(idx)], dtype=np.float32)
        score_type = _detect_score_type(scores)
        row_sum = float(np.sum(scores))

        return {
            "dataset": dataset,
            "idx": int(idx),
            "k": int(k),
            "mode": "precomputed_npy",
            "score_type": score_type,
            "row_sum": row_sum,
            "image_url": image_url,
            "topk": _topk_to_labels(dataset, scores, int(k), score_type),
        }

    if os.path.exists(paths["text_match_json"]):
        tmj = _read_json(paths["text_match_json"])
        # 支持两种格式：{"scores": [[...],[...]]} 或直接 [[...],[...]]
        scores_all = tmj.get("scores") if isinstance(tmj, dict) else tmj
        if not isinstance(scores_all, list):
            raise HTTPException(status_code=500, detail="text_match json invalid")
        if not (0 <= int(idx) < len(scores_all)):
            raise HTTPException(status_code=404, detail="idx out of range")
        scores = np.asarray(scores_all[int(idx)], dtype=np.float32)
        score_type = _detect_score_type(scores)
        return {
            "dataset": dataset,
            "idx": int(idx),
            "k": int(k),
            "mode": "precomputed_json",
            "score_type": score_type,
            "row_sum": float(np.sum(scores)),
            "image_url": image_url,
            "topk": _topk_to_labels(dataset, scores, int(k), score_type),
        }

    # 2) 降级：features + class centroids
    features = _load_features(dataset).astype(np.float32, copy=False)
    if not (0 <= int(idx) < features.shape[0]):
        raise HTTPException(status_code=404, detail="idx out of range")

    centroids = _get_normalized_centroids(dataset)
    f = np.asarray(features[int(idx)], dtype=np.float32)
    fn = float(np.linalg.norm(f))
    if fn < 1e-12:
        raise HTTPException(status_code=500, detail="zero-norm feature")
    f = f / fn
    scores = centroids @ f
    score_type = "score"

    return {
        "dataset": dataset,
        "idx": int(idx),
        "k": int(k),
        "mode": mode,
        "score_type": score_type,
        "row_sum": float(np.sum(scores)),
        "image_url": image_url,
        "topk": _topk_to_labels(dataset, scores, int(k), score_type),
    }


@app.get("/api/report/{dataset}/confusion-matrix")
def get_confusion_matrix(dataset: str):
    dataset = _ensure_dataset_name(dataset)
    data = _load_confusion_matrix(dataset)
    return {
        "dataset": dataset,
        "accuracy": float(data.get("accuracy", 0.0)),
        "confusion_matrix": data.get("confusion_matrix", []),
    }


@app.get("/api/report/summary")
def report_summary():
    domain_map = {
        "food101": "美食",
        "stl10": "工业",
        "eurosat": "遥感",
        "cifar100": "通用",
    }
    rows = []
    for name in DATASET_CONFIG.keys():
        try:
            cm = _load_confusion_matrix(name)
            top1 = float(cm.get("accuracy", 0.0))
        except HTTPException:
            top1 = None

        rows.append(
            {
                "dataset": name,
                "domain": domain_map.get(name, ""),
                "top1_acc": (None if top1 is None else round(top1, 6)),
                "notes": "",
            }
        )
    return {"rows": rows}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")