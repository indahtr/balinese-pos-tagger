from __future__ import annotations

import re
import unicodedata
from typing import List, Optional

_WS = re.compile(r"\s+")

_INVALID_TOKEN = re.compile(r"[^0-9A-Za-z.,!?;:()\"'\-%/@+]+")
_INVALID_TAG = re.compile(r"[^A-Z]+")
_PUNCT = re.compile(r"([,!?;:()\"'])")

_NORMALIZE_MAP = str.maketrans({
    "“": '"', "”": '"',   
    "‘": "'", "’": "'",   
    "–": "-", "—": "-",  
})

def remove_diacritics(text: Optional[str]) -> str:
    if text is None:
        return ""
    
    s = unicodedata.normalize("NFC", str(text))
    s = s.translate(_NORMALIZE_MAP)
    
    nfkd_form = unicodedata.normalize("NFKD", s)
    
    return "".join(ch for ch in nfkd_form if not unicodedata.combining(ch))

def normalize_whitespace(text: Optional[str]) -> str:
    s = str(text) if text is not None else ""
    return _WS.sub(" ", s).strip()

def clean_token(token: Optional[str]) -> str:
    s = remove_diacritics(token)
    
    s = normalize_whitespace(s)
    
    if not s:
        return ""
    
    return _INVALID_TOKEN.sub("", s).strip()

def clean_tag(tag: Optional[str]) -> str:
    if tag is None: 
        return ""
    s = remove_diacritics(tag).upper()
    return _INVALID_TAG.sub("", s).strip() if s else ""

def tokenize(text: Optional[str]) -> List[str]:
    s = remove_diacritics(text)
    if not s:
        return []

    # 1. Pisahkan tanda baca selain titik & koma
    s = re.sub(r"([!?;:()\"'])", r" \1 ", s)
    
    # 2. Handle titik (.) → jangan pecah angka
    s = re.sub(r"(?<!\d)\.(?!\d)", " . ", s)
    
    # 3. Handle koma (,) → jangan pecah angka desimal
    s = re.sub(r"(?<!\d),(?!\d)", " , ", s)
    
    # 4. Normalisasi spasi
    s = normalize_whitespace(s)
    
    tokens = s.split()
    
    # Step 4: Bersihkan tiap token secara individu
    return [clean_token(t) for t in tokens if clean_token(t)]