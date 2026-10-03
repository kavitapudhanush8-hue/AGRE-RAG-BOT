"""
Multilingual NLP Processing Pipeline for FarmAI.

Responsibilities:
1. Agriculture relevance detection
2. Dynamic crop detection (chilli, cotton, groundnut, rice, etc.)
3. Dynamic domain detection (pest, disease, irrigation, soil, etc.)
4. Query normalization & rewriting for retrieval
5. Query expansion & concept mapping across languages
6. Conversation context resolution

All functions are stateless and can be called independently.
Adding a new crop requires only extending the vocabulary dicts — no logic changes.
"""

import re
import logging

logger = logging.getLogger("farmai.nlp")

# ── Agriculture Relevance ────────────────────────────────────────────────────

AGRICULTURE_KEYWORDS = frozenset([
    # English
    "crop", "crops", "farm", "farming", "farmer", "agriculture", "agricultural",
    "soil", "seed", "seeds", "sow", "sowing", "transplant", "transplanting",
    "harvest", "harvesting", "irrigation", "irrigate", "water", "watering",
    "fertilizer", "fertilizers", "manure", "compost", "organic",
    "pest", "pests", "pesticide", "pesticides", "insect", "insects",
    "disease", "diseases", "fungicide", "herbicide", "weed", "weeds",
    "plant", "plants", "planting", "cultivate", "cultivation",
    "yield", "acre", "hectare", "field", "paddy",
    "nitrogen", "phosphorus", "potassium", "npk", "urea",
    "drip", "sprinkler", "mulching", "pruning", "grafting",
    "nursery", "seedling", "seedlings",
    # Crops (English)
    "rice", "wheat", "cotton", "chilli", "chili", "groundnut", "peanut",
    "tomato", "maize", "corn", "sugarcane", "soybean", "sunflower",
    "onion", "potato", "brinjal", "eggplant", "okra", "ladyfinger",
    "mango", "banana", "grape", "citrus", "guava", "papaya",
    "sorghum", "jowar", "bajra", "ragi", "turmeric", "mustard", "sesame",
    "horticulture", "floriculture", "sericulture", "apiculture",
    "agroforestry", "permaculture", "polyhouse", "greenhouse",
    "vermicompost", "biofertilizer", "biopesticide",
    "ph", "loam", "clay", "sandy", "laterite", "alluvial",
    "kharif", "rabi", "zaid",
    "tractor", "plough", "plow", "thresher", "combine",
    "storage", "godown", "warehouse", "msp",
    "nutrient", "deficiency", "yellowing", "wilting", "blight",
    "borer", "thrip", "thrips", "mite", "mites", "whitefly", "aphid",
    "rot", "mildew", "rust", "smut", "mosaic",
    "bollworm", "leafminer", "stemfly", "grub",
    # Hindi transliterated
    "kheti", "fasal", "kisan", "mitti", "beej", "ped", "paudha",
    "sinchai", "keet", "rog", "urvarak", "upaj", "krishi",
    "kapas", "dhan", "gehun", "makka", "moongfali",
    # Telugu transliterated
    "panta", "pantalu", "raithu", "matti", "vittanalu", "neeravari",
    "purugulu", "rogalu", "eruvu", "eruvulu", "sagu", "vyavasayam",
    "mirapakaya", "vari", "dhaanyam", "pandlu",
    "mirchi", "mirapa", "tomata", "bendakaya", "vankaya",
    "dondakaya", "beerakaya", "sorakaya", "kakarakaya",
    "gummadikaya", "pesara", "minumu", "kandulu",
    "senaga", "verusenaga", "jonna", "sajja", "korra",
    "patti", "goddi",  # cotton, wheat in Telugu
    # Hindi script
    "\u092e\u093f\u0930\u094d\u091a", "\u092e\u093f\u0930\u094d\u091a\u0940", "\u0916\u0947\u0924\u0940", "\u092b\u0938\u0932", "\u0915\u093f\u0938\u093e\u0928",
    "\u092e\u093f\u091f\u094d\u091f\u0940", "\u092c\u0940\u091c", "\u092a\u094c\u0927\u093e", "\u0938\u093f\u0902\u091a\u093e\u0908", "\u0915\u0940\u091f",
    "\u0930\u094b\u0917", "\u0909\u0930\u094d\u0935\u0930\u0915", "\u0909\u092a\u091c", "\u0915\u0943\u0937\u093f",
    "\u0927\u093e\u0928", "\u091a\u093e\u0935\u0932", "\u0917\u0947\u0939\u0942\u0902", "\u0915\u092a\u093e\u0938",
    "\u092a\u093e\u0928\u0940", "\u0916\u093e\u0926", "\u092b\u0942\u0932", "\u092b\u0932",
    "\u092a\u0924\u094d\u0924\u0940", "\u091f\u092e\u093e\u091f\u0930", "\u092c\u0948\u0902\u0917\u0928",
    "\u092a\u094d\u092f\u093e\u091c", "\u0906\u0932\u0942",
    "\u0915\u092a\u093e\u0938",  # kapas (cotton in Hindi)
    "\u0927\u093e\u0928",       # dhan (paddy/rice in Hindi)
    "\u092e\u0942\u0902\u0917\u092b\u0932\u0940",  # moongfali (groundnut)
    # Telugu script
    "\u0c2e\u0c3f\u0c30\u0c2a", "\u0c2e\u0c3f\u0c30\u0c2a\u0c15\u0c3e\u0c2f", "\u0c2a\u0c02\u0c1f", "\u0c28\u0c47\u0c32",
    "\u0c28\u0c40\u0c30\u0c41", "\u0c0e\u0c30\u0c41\u0c35\u0c41", "\u0c2a\u0c41\u0c30\u0c41\u0c17\u0c41",
    "\u0c30\u0c4b\u0c17\u0c02", "\u0c38\u0c3e\u0c17\u0c41", "\u0c35\u0c3f\u0c24\u0c4d\u0c24\u0c28\u0c3e\u0c32\u0c41",
    "\u0c35\u0c30\u0c3f", "\u0c27\u0c3e\u0c28\u0c4d\u0c2f\u0c02", "\u0c2a\u0c02\u0c21\u0c4d\u0c32\u0c41",
    "\u0c35\u0c4d\u0c2f\u0c35\u0c38\u0c3e\u0c2f\u0c02", "\u0c30\u0c48\u0c24\u0c41",
    "\u0c2e\u0c1f\u0c4d\u0c1f\u0c3f", "\u0c2c\u0c42\u0c2e\u0c3f",
    "\u0c2a\u0c24\u0c4d\u0c24\u0c3f",  # patti (cotton in Telugu)
])

AGRICULTURE_PHRASES = [
    "soil ph", "crop rotation", "drip irrigation", "organic farming",
    "plant growth", "weed management", "pest control", "disease control",
    "nutrient management", "water management", "post harvest",
    "seed treatment", "land preparation", "crop protection",
    "government scheme", "minimum support price", "crop insurance",
    "precision agriculture", "integrated pest management",
    "integrated nutrient management", "boll development", "boll rot",
    "leaf curl", "leaf spot", "stem rot", "root rot", "collar rot",
    "white grub", "leaf miner", "fruit borer", "pod borer",
]


def is_agriculture_related(text: str) -> bool:
    """Check whether text is related to agriculture/farming."""
    lower = text.lower()

    for phrase in AGRICULTURE_PHRASES:
        if phrase in lower:
            return True

    words = set(re.findall(r"[a-zA-Z\u0900-\u097F\u0C00-\u0C7F]+", lower))
    matches = words & AGRICULTURE_KEYWORDS
    if matches:
        return True

    return False


# ── Dynamic Crop Detection ───────────────────────────────────────────────────
# Maps crop names in all supported languages to a canonical English crop name.
# This allows the pipeline to understand which crop the user is asking about
# regardless of language. Extend this dict to add new crops.

CROP_VOCABULARY = {
    # Chilli
    "chilli": "chilli", "chili": "chilli", "chillies": "chilli",
    "pepper": "chilli", "capsicum": "chilli",
    "mirchi": "chilli", "mirch": "chilli", "mirapa": "chilli",
    "mirapakaya": "chilli", "mirapakayalu": "chilli",
    "\u092e\u093f\u0930\u094d\u091a": "chilli", "\u092e\u093f\u0930\u094d\u091a\u0940": "chilli",
    "\u0c2e\u0c3f\u0c30\u0c2a": "chilli", "\u0c2e\u0c3f\u0c30\u0c2a\u0c15\u0c3e\u0c2f": "chilli",
    # Cotton
    "cotton": "cotton", "kapas": "cotton", "patti": "cotton",
    "gossypium": "cotton", "bt cotton": "cotton",
    "\u0915\u092a\u093e\u0938": "cotton",
    "\u0c2a\u0c24\u0c4d\u0c24\u0c3f": "cotton",
    # Groundnut
    "groundnut": "groundnut", "peanut": "groundnut",
    "moongfali": "groundnut", "verusenaga": "groundnut",
    "palli": "groundnut", "senaga": "groundnut",
    "\u092e\u0942\u0902\u0917\u092b\u0932\u0940": "groundnut",
    "\u0c35\u0c47\u0c30\u0c41\u0c38\u0c46\u0c28\u0c17": "groundnut", "\u0c35\u0c47\u0c30\u0c41\u0c36\u0c46\u0c28\u0c17": "groundnut",
    # Rice
    "rice": "rice", "paddy": "rice", "dhan": "rice",
    "chawal": "rice", "vari": "rice", "biyyam": "rice",
    "oryza": "rice",
    "\u0927\u093e\u0928": "rice", "\u091a\u093e\u0935\u0932": "rice",
    "\u0c35\u0c30\u0c3f": "rice",
    # Tomato
    "tomato": "tomato", "tomata": "tomato", "tamatar": "tomato",
    "\u091f\u092e\u093e\u091f\u0930": "tomato",
    # Maize
    "maize": "maize", "corn": "maize", "makka": "maize",
    "jonna": "maize",
    "\u092e\u0915\u094d\u0915\u093e": "maize",
    # Wheat
    "wheat": "wheat", "gehun": "wheat", "goddi": "wheat", "godhuma": "wheat",
    "\u0917\u0947\u0939\u0942\u0902": "wheat",
    # Banana
    "banana": "banana", "kela": "banana", "arati": "banana",
    "\u0915\u0947\u0932\u093e": "banana",
    # Sugarcane
    "sugarcane": "sugarcane", "ganna": "sugarcane", "cheruku": "sugarcane",
    "\u0917\u0928\u094d\u0928\u093e": "sugarcane",
    # Soybean
    "soybean": "soybean", "soya": "soybean",
    # Sunflower
    "sunflower": "sunflower", "surajmukhi": "sunflower",
    # Onion
    "onion": "onion", "pyaz": "onion", "ullipaya": "onion",
    "ulli": "onion",
    # Potato
    "potato": "potato", "aloo": "potato", "bangaladumpa": "potato",
    # Brinjal/Eggplant
    "brinjal": "brinjal", "eggplant": "brinjal",
    "baingan": "brinjal", "vankaya": "brinjal",
    # Okra
    "okra": "okra", "ladyfinger": "okra", "bhindi": "okra",
    "bendakaya": "okra",
    # Mango
    "mango": "mango", "aam": "mango", "mamidi": "mango",
    # Turmeric
    "turmeric": "turmeric", "haldi": "turmeric", "pasupu": "turmeric",
    # Millets
    "sorghum": "sorghum", "jowar": "sorghum",
    "bajra": "pearl_millet", "sajja": "pearl_millet",
    "ragi": "finger_millet", "korra": "foxtail_millet",
    # Pulses
    "pesara": "green_gram", "minumu": "black_gram",
    "kandulu": "red_gram",
}


def detect_crop(text: str) -> str | None:
    """
    Detect the crop the user is asking about.

    Returns the canonical English crop name or None if no crop is detected.
    This works across all supported languages — English, Hindi, Telugu, Tenglish.
    """
    lower = text.lower()

    # Check multi-word crops first
    multi_word_crops = {"bt cotton": "cotton"}
    for phrase, crop in multi_word_crops.items():
        if phrase in lower:
            return crop

    # Tokenize and check each word
    words = re.findall(r"[a-zA-Z\u0900-\u097F\u0C00-\u0C7F]+", lower)
    for word in words:
        if word in CROP_VOCABULARY:
            return CROP_VOCABULARY[word]

    return None


# ── Dynamic Domain Detection ────────────────────────────────────────────────
# Maps keywords/phrases to canonical agriculture domains.

DOMAIN_VOCABULARY = {
    # Pest management
    "pest": "pest_management", "pests": "pest_management",
    "pesticide": "pest_management", "pesticides": "pest_management",
    "insect": "pest_management", "insects": "pest_management",
    "insecticide": "pest_management",
    "aphid": "pest_management", "aphids": "pest_management",
    "whitefly": "pest_management", "thrip": "pest_management",
    "thrips": "pest_management", "mite": "pest_management",
    "mites": "pest_management", "borer": "pest_management",
    "bollworm": "pest_management", "stemfly": "pest_management",
    "grub": "pest_management", "leafminer": "pest_management",
    "keet": "pest_management", "purugulu": "pest_management",
    "purugu": "pest_management",
    "\u0915\u0940\u091f": "pest_management",
    "\u0c2a\u0c41\u0c30\u0c41\u0c17\u0c41": "pest_management",
    # Crop diseases
    "disease": "crop_diseases", "diseases": "crop_diseases",
    "fungicide": "crop_diseases", "blight": "crop_diseases",
    "rot": "crop_diseases", "wilt": "crop_diseases",
    "wilting": "crop_diseases", "rust": "crop_diseases",
    "mildew": "crop_diseases", "mosaic": "crop_diseases",
    "yellowing": "crop_diseases", "yellow": "crop_diseases",
    "leaf spot": "crop_diseases", "smut": "crop_diseases",
    "rog": "crop_diseases", "rogalu": "crop_diseases",
    "rogam": "crop_diseases",
    "\u0930\u094b\u0917": "crop_diseases",
    "\u0c30\u0c4b\u0c17\u0c02": "crop_diseases",
    # Irrigation
    "irrigation": "irrigation", "irrigate": "irrigation",
    "water": "irrigation", "watering": "irrigation",
    "drip": "irrigation", "sprinkler": "irrigation",
    "sinchai": "irrigation", "neeravari": "irrigation",
    "neeru": "irrigation", "neellu": "irrigation",
    "\u0938\u093f\u0902\u091a\u093e\u0908": "irrigation",
    "\u0c28\u0c40\u0c30\u0c41": "irrigation",
    # Soil management
    "soil": "soil_management", "mitti": "soil_management",
    "nela": "soil_management", "matti": "soil_management",
    "ph": "soil_management", "loam": "soil_management",
    "clay": "soil_management", "sandy": "soil_management",
    "\u092e\u093f\u091f\u094d\u091f\u0940": "soil_management",
    "\u0c28\u0c47\u0c32": "soil_management",
    # Fertilizer
    "fertilizer": "fertilizer_management",
    "fertilizers": "fertilizer_management",
    "manure": "fertilizer_management", "compost": "fertilizer_management",
    "npk": "fertilizer_management", "urea": "fertilizer_management",
    "nitrogen": "fertilizer_management", "phosphorus": "fertilizer_management",
    "potassium": "fertilizer_management", "nutrient": "fertilizer_management",
    "deficiency": "fertilizer_management",
    "eruvu": "fertilizer_management", "eruvulu": "fertilizer_management",
    "urvarak": "fertilizer_management", "khad": "fertilizer_management",
    "\u0909\u0930\u094d\u0935\u0930\u0915": "fertilizer_management",
    "\u0c0e\u0c30\u0c41\u0c35\u0c41": "fertilizer_management",
    # Seed & sowing
    "seed": "seed_selection", "seeds": "seed_selection",
    "sow": "sowing", "sowing": "sowing",
    "beej": "seed_selection", "vittanalu": "seed_selection",
    "\u092c\u0940\u091c": "seed_selection",
    "\u0c35\u0c3f\u0c24\u0c4d\u0c24\u0c28\u0c3e\u0c32\u0c41": "seed_selection",
    # Transplanting / nursery
    "transplant": "transplanting", "transplanting": "transplanting",
    "nursery": "nursery_management",
    "seedling": "nursery_management", "seedlings": "nursery_management",
    # Harvesting
    "harvest": "harvesting", "harvesting": "harvesting",
    "kapu": "harvesting",
    # Weed management
    "weed": "weed_management", "weeds": "weed_management",
    "herbicide": "weed_management",
    # General cultivation
    "cultivation": "cultivation", "cultivate": "cultivation",
    "farming": "cultivation", "growing": "cultivation",
    "sagu": "cultivation",
    "\u0c38\u0c3e\u0c17\u0c41": "cultivation",
}

# Multi-word domain phrases (checked as substrings)
DOMAIN_PHRASES = {
    "pest control": "pest_management",
    "pest management": "pest_management",
    "integrated pest": "pest_management",
    "disease control": "crop_diseases",
    "leaf curl": "crop_diseases",
    "leaf spot": "crop_diseases",
    "stem rot": "crop_diseases",
    "root rot": "crop_diseases",
    "boll rot": "crop_diseases",
    "collar rot": "crop_diseases",
    "drip irrigation": "irrigation",
    "sprinkler irrigation": "irrigation",
    "water management": "irrigation",
    "water requirement": "irrigation",
    "soil ph": "soil_management",
    "soil type": "soil_management",
    "soil requirement": "soil_management",
    "organic farming": "organic_farming",
    "crop rotation": "cultivation",
    "weed management": "weed_management",
    "weed control": "weed_management",
    "seed treatment": "seed_selection",
    "seed selection": "seed_selection",
    "post harvest": "post_harvest",
    "nutrient management": "fertilizer_management",
    "nutrient deficiency": "fertilizer_management",
}


def detect_domain(text: str) -> str | None:
    """
    Detect the agriculture domain the user is asking about.
    Returns canonical domain name or None if unclear.
    """
    lower = text.lower()

    # Check phrases first (more specific)
    for phrase, domain in DOMAIN_PHRASES.items():
        if phrase in lower:
            return domain

    # Check individual keywords
    words = re.findall(r"[a-zA-Z\u0900-\u097F\u0C00-\u0C7F]+", lower)
    for word in words:
        if word in DOMAIN_VOCABULARY:
            return DOMAIN_VOCABULARY[word]

    return None


# ── Cross-language concept mapping ───────────────────────────────────────────

CONCEPT_MAP = {
    # Telugu transliteration -> English (multi-crop)
    "mirapakaya": "chilli", "mirapakayalu": "chilli", "mirapa": "chilli",
    "mirchi": "chilli", "mirch": "chilli",
    "vari": "rice", "varipanta": "rice", "biyyam": "rice",
    "panta": "crop", "pantalu": "crops",
    "sagu": "cultivation", "sagucheyali": "cultivation",
    "pandinchadam": "growing", "pandinchadaniki": "growing",
    "nela": "soil", "matti": "soil", "bhoomi": "land", "bhumi": "land",
    "neeravari": "irrigation", "neetipaaruvu": "irrigation",
    "neeru": "water", "neellu": "water", "neetiparuvu": "water supply",
    "eruvu": "fertilizer", "eruvulu": "fertilizers",
    "kompa": "manure",
    "purugulu": "pests", "purugu": "pest",
    "rogalu": "diseases", "rogam": "disease",
    "vittanalu": "seeds", "vitta": "seed", "vithanalu": "seeds",
    "aaku": "leaf", "aakulu": "leaves",
    "puvvu": "flower", "puvvulu": "flowers",
    "pandu": "fruit", "pandlu": "fruits",
    "gutti": "seed",
    "dhaanyam": "grain", "dhaanyalu": "grains",
    "raithu": "farmer", "raitu": "farmer",
    "vyavasayam": "agriculture",
    "kapu": "harvest", "kappu": "harvest",
    "patti": "cotton", "goddi": "wheat", "godhuma": "wheat",
    # Crops in Telugu transliteration
    "tomata": "tomato",
    "bendakaya": "okra", "vankaya": "brinjal",
    "dondakaya": "tinda", "beerakaya": "ridge gourd",
    "sorakaya": "bottle gourd", "gummadikaya": "pumpkin",
    "kakarakaya": "bitter gourd", "kakara": "bitter gourd",
    "pesara": "green gram", "minumu": "black gram",
    "kandulu": "red gram", "senaga": "chickpea",
    "verusenaga": "groundnut", "palli": "groundnut",
    "jonna": "sorghum", "sajja": "pearl millet",
    "korra": "foxtail millet",
    "nuvvulu": "sesame", "yellu": "sesame",
    "mandu": "pesticide", "mandulu": "pesticides",
    "gaddi": "grass",
    "ullipaya": "onion", "ulli": "onion",
    "bangaladumpa": "potato",
    "mamidi": "mango", "arati": "banana",
    "pasupu": "turmeric", "cheruku": "sugarcane",
    # Telugu verb forms
    "ivvali": "apply", "veyali": "apply",
    "cheyyali": "do", "chesukovali": "need to do",
    "vastunnayi": "coming", "avutundi": "happening",
    "undali": "should be", "untundi": "will be",
    # Hindi -> English
    "kheti": "farming", "fasal": "crop", "kisan": "farmer",
    "mitti": "soil", "beej": "seed", "paudha": "plant",
    "sinchai": "irrigation", "keet": "pest", "rog": "disease",
    "urvarak": "fertilizer", "upaj": "yield", "krishi": "agriculture",
    "kapas": "cotton", "dhan": "rice", "chawal": "rice",
    "gehun": "wheat", "makka": "maize",
    "moongfali": "groundnut", "tamatar": "tomato",
    "kela": "banana", "ganna": "sugarcane",
    "pyaz": "onion", "aloo": "potato", "baingan": "brinjal",
    "haldi": "turmeric",
    # Hindi script -> English
    "\u092e\u093f\u0930\u094d\u091a": "chilli", "\u092e\u093f\u0930\u094d\u091a\u0940": "chilli",
    "\u091a\u093e\u0935\u0932": "rice", "\u0927\u093e\u0928": "rice",
    "\u092e\u093f\u091f\u094d\u091f\u0940": "soil", "\u0916\u0947\u0924\u0940": "farming",
    "\u092b\u0938\u0932": "crop",
    "\u0938\u093f\u0902\u091a\u093e\u0908": "irrigation", "\u0915\u0940\u091f": "pest",
    "\u092c\u0940\u091c": "seed", "\u0909\u0930\u094d\u0935\u0930\u0915": "fertilizer",
    "\u0930\u094b\u0917": "disease",
    "\u092a\u093e\u0928\u0940": "water", "\u0916\u093e\u0926": "manure",
    "\u0915\u092a\u093e\u0938": "cotton",
    "\u092e\u0942\u0902\u0917\u092b\u0932\u0940": "groundnut",
    "\u0917\u0947\u0939\u0942\u0902": "wheat",
    "\u092e\u0915\u094d\u0915\u093e": "maize",
    # Telugu script -> English
    "\u0c2e\u0c3f\u0c30\u0c2a": "chilli", "\u0c2e\u0c3f\u0c30\u0c2a\u0c15\u0c3e\u0c2f": "chilli",
    "\u0c35\u0c30\u0c3f": "rice", "\u0c2a\u0c02\u0c1f": "crop", "\u0c28\u0c47\u0c32": "soil",
    "\u0c28\u0c40\u0c30\u0c41": "water", "\u0c0e\u0c30\u0c41\u0c35\u0c41": "fertilizer",
    "\u0c2a\u0c41\u0c30\u0c41\u0c17\u0c41": "pest", "\u0c30\u0c4b\u0c17\u0c02": "disease",
    "\u0c38\u0c3e\u0c17\u0c41": "cultivation", "\u0c35\u0c3f\u0c24\u0c4d\u0c24\u0c28\u0c3e\u0c32\u0c41": "seeds",
    "\u0c2a\u0c24\u0c4d\u0c24\u0c3f": "cotton",
}


def normalize_query_for_retrieval(text: str, language: str) -> str:
    """
    Produce an English-normalized version for TF-IDF retrieval.
    Maps all languages to English tokens.
    """
    if language == "ENGLISH":
        return _clean_text(text)

    tokens = re.findall(r"[\w\u0900-\u097F\u0C00-\u0C7F]+", text.lower())
    english_tokens = []

    for token in tokens:
        mapped = CONCEPT_MAP.get(token)
        if mapped:
            english_tokens.append(mapped)
        elif re.match(r"^[a-zA-Z]+$", token):
            english_tokens.append(token)

    normalized = " ".join(english_tokens)

    if len(english_tokens) < 3:
        mapped_extra = []
        for token in tokens:
            m = CONCEPT_MAP.get(token)
            if m and m not in english_tokens:
                mapped_extra.append(m)
        normalized = _clean_text(text) + " " + " ".join(mapped_extra)

    return normalized.strip()


def expand_query(normalized_query: str) -> str:
    """Expand with related agricultural terms for better retrieval recall."""
    expansions = {
        "chilli": "chilli chili pepper capsicum",
        "rice": "rice paddy oryza",
        "cotton": "cotton gossypium bt",
        "groundnut": "groundnut peanut arachis",
        "tomato": "tomato solanum lycopersicon",
        "maize": "maize corn",
        "wheat": "wheat triticum",
        "soybean": "soybean soya glycine",
        "irrigation": "irrigation water watering drip sprinkler",
        "fertilizer": "fertilizer fertiliser manure nutrient npk urea",
        "pest": "pest insect borer thrip mite whitefly aphid bollworm",
        "disease": "disease blight rot wilt mildew rust fungus fungal bacterial",
        "soil": "soil ph loam clay sandy laterite",
        "seed": "seed seeds sowing nursery seedling",
        "harvest": "harvest harvesting post-harvest storage yield",
        "okra": "okra ladyfinger bhindi bendakaya",
        "brinjal": "brinjal eggplant aubergine vankaya",
        "banana": "banana plantain",
        "sugarcane": "sugarcane sugar cane",
    }

    lower = normalized_query.lower()
    extra = []
    for key, expansion in expansions.items():
        if key in lower:
            extra.append(expansion)

    if extra:
        return normalized_query + " " + " ".join(extra)
    return normalized_query


def resolve_conversation_context(
    question: str,
    history: list[dict],
    max_turns: int = 5,
) -> str:
    """Resolve pronouns and references using recent conversation history."""
    if not history:
        return question

    reference_patterns = [
        r"\b(it|its|this|that|these|those|the same)\b",
        r"\b(this crop|that crop|the crop|this plant|that plant)\b",
        r"\b(this disease|that disease|this pest|that pest)\b",
        r"\b(above|mentioned|previous)\b",
        r"\b(adi|avi|ee|aa|ide|daani|vaati)\b",
    ]

    has_reference = any(
        re.search(pat, question, re.IGNORECASE) for pat in reference_patterns
    )

    if not has_reference:
        return question

    recent = history[-max_turns * 2:]
    context_parts = []
    for msg in recent:
        if msg.get("role") == "user":
            context_parts.append(f"User previously asked: {msg['content']}")
        elif msg.get("role") == "assistant":
            first_sentence = msg["content"].split(".")[0] + "."
            context_parts.append(f"Assistant answered about: {first_sentence}")

    context_summary = " | ".join(context_parts[-4:])
    enhanced = f"[Context: {context_summary}] Current question: {question}"
    return enhanced


def _clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Non-agriculture response templates ───────────────────────────────────────

NON_AGRICULTURE_RESPONSES = {
    "ENGLISH": (
        "I'm an agriculture-focused assistant, so I can help with farming, "
        "crops, soil, irrigation, pests, diseases, fertilizers, and related "
        "agricultural topics. Please feel free to ask me anything about agriculture!"
    ),
    "HINDI": (
        "\u092e\u0948\u0902 \u090f\u0915 \u0915\u0943\u0937\u093f-\u0915\u0947\u0902\u0926\u094d\u0930\u093f\u0924 \u0938\u0939\u093e\u092f\u0915 \u0939\u0942\u0901, \u0907\u0938\u0932\u093f\u090f \u092e\u0948\u0902 \u0916\u0947\u0924\u0940, \u092b\u0938\u0932\u094b\u0902, \u092e\u093f\u091f\u094d\u091f\u0940, "
        "\u0938\u093f\u0902\u091a\u093e\u0908, \u0915\u0940\u091f\u094b\u0902, \u0930\u094b\u0917\u094b\u0902, \u0909\u0930\u094d\u0935\u0930\u0915\u094b\u0902 \u0914\u0930 \u0938\u0902\u092c\u0902\u0927\u093f\u0924 \u0915\u0943\u0937\u093f \u0935\u093f\u0937\u092f\u094b\u0902 \u092e\u0947\u0902 \u0906\u092a\u0915\u0940 "
        "\u092e\u0926\u0926 \u0915\u0930 \u0938\u0915\u0924\u093e \u0939\u0942\u0901\u0964 \u0915\u0943\u092a\u092f\u093e \u0915\u0943\u0937\u093f \u0938\u0947 \u0938\u0902\u092c\u0902\u0927\u093f\u0924 \u0915\u094b\u0908 \u092d\u0940 \u092a\u094d\u0930\u0936\u094d\u0928 \u092a\u0942\u091b\u0947\u0902!"
    ),
    "TELUGU": (
        "\u0c28\u0c47\u0c28\u0c41 \u0c35\u0c4d\u0c2f\u0c35\u0c38\u0c3e\u0c2f-\u0c15\u0c47\u0c02\u0c26\u0c4d\u0c30\u0c3f\u0c24 \u0c38\u0c39\u0c3e\u0c2f\u0c15\u0c41\u0c21\u0c3f\u0c28\u0c3f, \u0c15\u0c3e\u0c2c\u0c1f\u0c4d\u0c1f\u0c3f \u0c28\u0c47\u0c28\u0c41 \u0c35\u0c4d\u0c2f\u0c35\u0c38\u0c3e\u0c2f\u0c02, "
        "\u0c2a\u0c02\u0c1f\u0c32\u0c41, \u0c28\u0c47\u0c32, \u0c28\u0c40\u0c1f\u0c3f\u0c2a\u0c3e\u0c30\u0c41\u0c26\u0c32, \u0c24\u0c46\u0c17\u0c41\u0c33\u0c4d\u0c33\u0c41, \u0c35\u0c4d\u0c2f\u0c3e\u0c27\u0c41\u0c32\u0c41, \u0c0e\u0c30\u0c41\u0c35\u0c41\u0c32\u0c41 \u0c2e\u0c30\u0c3f\u0c2f\u0c41 "
        "\u0c38\u0c02\u0c2c\u0c02\u0c27\u0c3f\u0c24 \u0c35\u0c4d\u0c2f\u0c35\u0c38\u0c3e\u0c2f \u0c05\u0c02\u0c36\u0c3e\u0c32\u0c32\u0c4b \u0c2e\u0c40\u0c15\u0c41 \u0c38\u0c39\u0c3e\u0c2f\u0c02 \u0c1a\u0c47\u0c2f\u0c17\u0c32\u0c28\u0c41. \u0c26\u0c2f\u0c1a\u0c47\u0c38\u0c3f \u0c35\u0c4d\u0c2f\u0c35\u0c38\u0c3e\u0c2f\u0c3e\u0c28\u0c3f\u0c15\u0c3f "
        "\u0c38\u0c02\u0c2c\u0c02\u0c27\u0c3f\u0c02\u0c1a\u0c3f\u0c28 \u0c0f\u0c26\u0c48\u0c28\u0c3e \u0c2a\u0c4d\u0c30\u0c36\u0c4d\u0c28 \u0c05\u0c21\u0c17\u0c02\u0c21\u0c3f!"
    ),
    "TENGLISH": (
        "Nenu oka agriculture-focused assistant ni, kabatti farming, crops, "
        "soil, irrigation, pests, diseases, fertilizers mariyu related "
        "agricultural topics lo meeku help cheyagalanu. Dayachesi agriculture "
        "gurinchi emaina question adagandi!"
    ),
}


def get_non_agriculture_response(language: str) -> str:
    """Return the non-agriculture redirect message in the given language."""
    return NON_AGRICULTURE_RESPONSES.get(language, NON_AGRICULTURE_RESPONSES["ENGLISH"])
