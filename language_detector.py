"""
Language detection module for FarmAI multilingual chatbot.

Detects: ENGLISH, HINDI, TELUGU, TENGLISH

Detection strategy (in order):
1. Unicode script analysis — Hindi (Devanagari) and Telugu script are
   detected by character ranges.  If a majority of alphabetic chars
   belong to one script, the decision is immediate.
2. Tenglish heuristic — for Latin-script text, check for Telugu
   transliterated words / sentence patterns mixed with English.
3. Default — pure English.
"""

import re
import logging

logger = logging.getLogger("farmai.language")

# ── Telugu transliterated vocabulary ─────────────────────────────────────────
# Common Telugu words written in Roman script (agriculture-relevant + general).
# Kept as a frozenset for O(1) lookups.
TELUGU_ROMAN_WORDS = frozenset([
    # General Telugu words
    "enti", "entha", "ela", "enduku", "eppudu", "ekkada", "evaru",
    "edi", "em", "entha", "anni", "inka", "kuda", "kani", "ante",
    "aythe", "ledu", "undi", "untundi", "cheyyali", "cheppandi",
    "cheyandi", "kavali", "avunu", "kaadu", "meeru", "nenu", "maaku",
    "naaku", "vaalu", "vaallaki", "daani", "daanini", "ikkada",
    "akkada", "appudu", "ippudu", "mundu", "tarvata", "kotha",
    "patha", "manchi", "cheddadi", "pedda", "chinna", "ekkuva",
    "takkuva", "lo", "ki", "ni", "tho", "leka", "lanti", "ga",
    "ku", "kosam", "gurinchi", "valla", "dwara", "paina", "kinda",
    "madhya", "daggaraga", "dooram", "okatey", "rendu", "moodu",
    "nalugu", "aidu", "aaru", "yedu", "yenimidi", "tommidi", "padi",
    "vanda", "veyi", "laksham",
    # Agriculture-specific Telugu transliterations
    "mirapakaya", "mirapakayalu", "mirapa", "mirchi",
    "vari", "varipanta", "panta", "pantalu", "sagucheyali",
    "sagu", "sagucheyyadam", "pandinchadaniki", "pandinchadam",
    "nela", "bhumi", "neetiparuvu", "neetipaaruvu",
    "eruvu", "eruvulu", "kompa", "ettadam",
    "purugulu", "rogalu", "rogam",
    "neeru", "neeravarulu", "neeravari", "chedumedam",
    "vittanalu", "vitta", "vithana", "vithanalu",
    "kotha", "pungatipanta", "puradom",
    "dhaanyam", "dhaanyalu",
    "pandlu", "kaapu", "kappu", "kapu",
    "puvvu", "puvvulu", "aaku", "aakulu",
    "korra", "sajja", "jonna", "ragi", "pesara", "minumu",
    "kandulu", "senaga", "verusenaga",
    "tomato", "tomata", "dondakaya", "bendakaya", "beerakaya",
    "sorakaya", "gummadikaya", "potlakaya", "kakara", "kakarakaya",
    "vankaya", "gutti", "pandu", "kayalu",
    "paduchuheyyi", "paduchuheyaali",
    "chesukovali", "ivvali", "ivvandi",
    "veyali", "veyandi", "kottali",
    "vastunnayi", "vachche", "avutundi", "avutunnayi",
    "cheyyali", "cheyyandi", "chesukovali",
    "yellu", "nuvvulu", "mandu", "mandulu",
    "panchayiti", "vyavasayam", "raitu", "raithu",
    "bhoomi", "matti", "gaddi",
    "panchayati", "paddakam",
    "unna", "unde", "undi", "unnayi",
    "chesthe", "cheste", "ante", "ayithe",
    "dorukutundi", "dorukutunnadi",
    "tayyaari", "tayaaree",
])

# Patterns that indicate Tenglish sentence structure:
# Telugu postpositions / verb endings appearing after English or Roman words.
TENGLISH_PATTERNS = [
    # verb endings
    r"\b\w+ali\b",          # cheyyali, ivvali, veyali
    r"\b\w+andi\b",         # cheppandi, ivvandi
    r"\b\w+undi\b",         # untundi, avutundi
    r"\b\w+tunnayi\b",      # vastunnayi, avutunnayi
    r"\b\w+tundi\b",        # dorukutundi
    r"\b\w+dam\b",          # pandinchadam, cheyadam
    r"\b\w+tham\b",         # chesutham
    # postpositions attached or standalone
    r"\b\w+ki\b",           # crop ki, panta ki
    r"\b\w+lo\b",           # crop lo, nela lo
    r"\b\w+ku\b",           # daaniku
    r"\b\w+tho\b",          # neetitho
    r"\b\w+kosam\b",        # daani kosam
]

# Devanagari Unicode range (Hindi)
DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")
# Telugu Unicode range
TELUGU_RANGE = re.compile(r"[\u0C00-\u0C7F]")
# Latin letters
LATIN_RANGE = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """
    Detect the language of *text* and return one of:
    ``"ENGLISH"``, ``"HINDI"``, ``"TELUGU"``, ``"TENGLISH"``.

    The function is intentionally lightweight (no ML model, no external
    API) so it adds near-zero latency to every chat request.
    """
    if not text or not text.strip():
        return "ENGLISH"

    text = text.strip()

    # ── Step 1: Unicode script detection ─────────────────────────────
    devanagari_chars = len(DEVANAGARI_RANGE.findall(text))
    telugu_chars = len(TELUGU_RANGE.findall(text))
    latin_chars = len(LATIN_RANGE.findall(text))
    total_alpha = devanagari_chars + telugu_chars + latin_chars

    if total_alpha == 0:
        return "ENGLISH"

    # If ≥30% of alphabetic characters are Devanagari → Hindi
    if devanagari_chars / total_alpha >= 0.30:
        logger.debug("Detected HINDI (%.0f%% Devanagari chars)", devanagari_chars / total_alpha * 100)
        return "HINDI"

    # If ≥30% of alphabetic characters are Telugu script → Telugu
    if telugu_chars / total_alpha >= 0.30:
        logger.debug("Detected TELUGU (%.0f%% Telugu chars)", telugu_chars / total_alpha * 100)
        return "TELUGU"

    # ── Step 2: Tenglish detection (Latin-only text) ─────────────────
    if latin_chars > 0:
        words = re.findall(r"[A-Za-z]+", text.lower())
        if words:
            telugu_word_count = sum(1 for w in words if w in TELUGU_ROMAN_WORDS)
            telugu_ratio = telugu_word_count / len(words)

            # Also check structural patterns
            pattern_hits = sum(
                1 for pat in TENGLISH_PATTERNS
                if re.search(pat, text, re.IGNORECASE)
            )

            # Decision: if ≥15% of words are Telugu transliterations, or
            # ≥2 Tenglish patterns fire and at least 1 Telugu word exists
            if telugu_ratio >= 0.15:
                logger.debug(
                    "Detected TENGLISH (%.0f%% Telugu words, %d pattern hits)",
                    telugu_ratio * 100, pattern_hits,
                )
                return "TENGLISH"

            if pattern_hits >= 2 and telugu_word_count >= 1:
                logger.debug(
                    "Detected TENGLISH (pattern-based: %d hits, %d Telugu words)",
                    pattern_hits, telugu_word_count,
                )
                return "TENGLISH"

            # Specific agriculture-related Tenglish patterns
            agri_tenglish = re.search(
                r"\b(panta|saguchey|pandinch|mirapa|vari|eruvu|purugu|rogu|neeru|vittanalu|raithu|vyavasayam|matti)\w*\b",
                text, re.IGNORECASE,
            )
            if agri_tenglish:
                logger.debug("Detected TENGLISH (agriculture-specific word: %s)", agri_tenglish.group())
                return "TENGLISH"

    # ── Step 3: Default to English ───────────────────────────────────
    logger.debug("Detected ENGLISH (default)")
    return "ENGLISH"


if __name__ == "__main__":
    # Quick test
    tests = [
        ("What is the best soil pH for chilli?", "ENGLISH"),
        ("मिर्च की खेती के लिए मिट्टी का pH कितना होना चाहिए?", "HINDI"),
        ("మిరప పంటకు నేల pH ఎంత ఉండాలి?", "TELUGU"),
        ("Mirapakaya crop ki soil pH entha undali?", "TENGLISH"),
        ("Chilli crop ki irrigation ela manage cheyyali?", "TENGLISH"),
        ("Tomato crop lo yellow leaves enduku vastunnayi?", "TENGLISH"),
        ("Who is the president of the United States?", "ENGLISH"),
    ]
    for text, expected in tests:
        result = detect_language(text)
        status = "✓" if result == expected else "✗"
        print(f"{status}  [{result:8s}] (expected {expected:8s})  {text}")
