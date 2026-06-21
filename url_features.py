"""
url_features.py — Ekstraksi fitur LEKSIKAL dari sebuah URL.

PENTING (anti train-serving skew):
    Modul ini adalah SATU-SATUNYA sumber kebenaran untuk fitur.
    Notebook (skripsi.ipynb) memakai fungsi ini untuk membangun X dari kolom 'URL',
    dan backend (app.py) memakai fungsi YANG SAMA saat inference.
    Karena fungsi & urutan kolomnya identik, nilai fitur saat training == saat produksi.

Desain anti-leakage:
    - TIDAK memakai fitur konten/DOM (NoOfCSS, HasCopyrightInfo, dll) -> leakage PhiUSIIL.
    - TIDAK memakai fitur target-encoding (URLSimilarityIndex, TLDLegitimateProb).
    - Semua fitur deterministik dari string URL (bisa dihitung tanpa men-scrape).
    - Daftar kata/TLD mencurigakan = pengetahuan domain (bukan diturunkan dari label dataset).
"""

import re
import math
from urllib.parse import urlparse

# Kata yang sering muncul di URL phishing (sinyal leksikal, bukan leakage label).
SUSPICIOUS_WORDS = (
    "login", "signin", "sign-in", "verify", "verification", "account", "update",
    "secure", "security", "bank", "paypal", "confirm", "password", "webscr",
    "ebayisapi", "wp-admin", "free", "bonus", "gift", "alert", "support",
    "invoice", "billing", "wallet", "recover", "unlock", "auth", "session",
    "validate", "suspended", "limited", "unusual", "activity",
)

# TLD yang relatif sering dipakai phishing / gratisan (pengetahuan umum, bukan dari label).
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "online", "site",
    "work", "live", "click", "link", "zip", "review", "country", "kim",
    "science", "party", "gdn", "stream", "date", "loan", "men", "racing",
}

VOWELS = set("aeiou")

# Karakter tanda baca yang "wajar" ada di URL (RFC 3986). Selain ini = karakter aneh.
_ALLOWED_PUNCT = set("-._~:/?#[]@!$&'()*+,;=%")

# Urutan kolom WAJIB identik antara training dan inference.
FEATURE_ORDER = [
    # --- panjang / ukuran ---
    "URLLength", "DomainLength", "TLDLength", "PathLength", "QueryLength",
    "NoOfSubDomain", "SubdomainLength", "LongestTokenLength", "NoOfTokens",
    # --- komposisi karakter ---
    "NoOfLettersInURL", "LetterRatioInURL",
    "NoOfDigitsInURL", "DigitRatioInURL",
    "NoOfDigitsInDomain", "DomainDigitRatio",
    "VowelRatioInURL", "URLEntropy", "DomainEntropy", "CharContinuationRate",
    # --- karakter spesial ---
    "NoOfDotsInURL", "NoOfHyphensInURL", "NoOfHyphensInDomain", "NoOfSlashInPath",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL", "NoOfPercentInURL",
    "NoOfOtherSpecialCharsInURL", "SpecialCharRatioInURL", "NoOfParams",
    # --- boolean / heuristik ---
    "IsDomainIP", "IsPortInURL", "HasAtSymbol", "HasDoubleSlashInPath",
    "HasHexEncoding", "IsPunycode", "HasHttpsTokenInHostPath",
    "HasObfuscation", "HasSuspiciousWord", "SuspiciousWordCount", "IsSuspiciousTLD",
]

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
_PCT_RE = re.compile(r"%[0-9a-fA-F]{2}")
_TOKEN_RE = re.compile(r"[\W_]+")


def normalize_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    if not _SCHEME_RE.match(raw):
        raw = "https://" + raw
    parsed = urlparse(raw)
    if parsed.netloc.lower().startswith("www."):
        raw = parsed._replace(netloc=parsed.netloc[4:]).geturl()
    # Buang trailing slash supaya 'x.com' == 'x.com/' dan 'x.com/a/' == 'x.com/a'
    # (slash di belakang tak bermakna; ini bikin claude.ai dan claude.ai/ identik).
    raw = re.sub(r"/+$", "", raw)
    return raw


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def extract_url_features(url: str) -> dict:
    """Kembalikan dict fitur leksikal. Key-nya == FEATURE_ORDER."""
    raw = normalize_url(url)
    parsed = urlparse(raw)

    netloc = parsed.netloc.lower()
    hostname = netloc.split(":")[0]
    path = parsed.path or ""
    query = parsed.query or ""

    n = max(len(raw), 1)
    low = raw.lower()

    letters = sum(c.isalpha() for c in raw)
    digits = sum(c.isdigit() for c in raw)
    vowels = sum(1 for c in low if c in VOWELS)

    # Struktur domain
    host_no_www = hostname[4:] if hostname.startswith("www.") else hostname
    host_parts = [p for p in host_no_www.split(".") if p]
    tld = host_parts[-1] if host_parts else ""
    no_subdomain = max(0, len(host_parts) - 2)
    subdomain = ".".join(host_parts[:-2]) if len(host_parts) > 2 else ""
    digits_in_domain = sum(c.isdigit() for c in hostname)

    # Token (kata) di seluruh URL
    tokens = [t for t in _TOKEN_RE.split(raw) if t]
    longest_token = max((len(t) for t in tokens), default=0)

    is_ip = int(bool(_IP_RE.match(hostname)))
    try:
        has_port = int(parsed.port is not None)
    except ValueError:
        has_port = 0

    other_special = sum(1 for c in raw if (not c.isalnum()) and (c not in _ALLOWED_PUNCT))
    special_total = sum(1 for c in raw if not c.isalnum())
    cont = sum(1 for i in range(1, len(raw)) if raw[i] == raw[i - 1])

    susp_count = sum(low.count(w) for w in SUSPICIOUS_WORDS)
    has_double_slash = int("//" in raw[8:])              # // setelah 'https://'
    has_hex = int(bool(_PCT_RE.search(raw)))
    has_at = int("@" in raw)
    has_obf = int(has_at or has_double_slash or has_hex)
    has_https_token = int(("https" in hostname) or ("https" in path.lower()))

    feats = {
        # panjang / ukuran
        "URLLength": len(raw),
        "DomainLength": len(hostname),
        "TLDLength": len(tld),
        "PathLength": len(path),
        "QueryLength": len(query),
        "NoOfSubDomain": no_subdomain,
        "SubdomainLength": len(subdomain),
        "LongestTokenLength": longest_token,
        "NoOfTokens": len(tokens),
        # komposisi karakter
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": round(letters / n, 6),
        "NoOfDigitsInURL": digits,
        "DigitRatioInURL": round(digits / n, 6),
        "NoOfDigitsInDomain": digits_in_domain,
        "DomainDigitRatio": round(digits_in_domain / max(len(hostname), 1), 6),
        "VowelRatioInURL": round(vowels / max(letters, 1), 6),
        "URLEntropy": round(_shannon_entropy(raw), 6),
        "DomainEntropy": round(_shannon_entropy(hostname), 6),
        "CharContinuationRate": round(cont / n, 6),
        # karakter spesial
        "NoOfDotsInURL": raw.count("."),
        "NoOfHyphensInURL": raw.count("-"),
        "NoOfHyphensInDomain": hostname.count("-"),
        "NoOfSlashInPath": path.count("/"),
        "NoOfEqualsInURL": raw.count("="),
        "NoOfQMarkInURL": raw.count("?"),
        "NoOfAmpersandInURL": raw.count("&"),
        "NoOfPercentInURL": raw.count("%"),
        "NoOfOtherSpecialCharsInURL": other_special,
        "SpecialCharRatioInURL": round(special_total / n, 6),
        "NoOfParams": (query.count("&") + 1) if query else 0,
        # boolean / heuristik
        "IsDomainIP": is_ip,
        "IsPortInURL": has_port,
        "HasAtSymbol": has_at,
        "HasDoubleSlashInPath": has_double_slash,
        "HasHexEncoding": has_hex,
        "IsPunycode": int("xn--" in hostname),
        "HasHttpsTokenInHostPath": has_https_token,
        "HasObfuscation": has_obf,
        "HasSuspiciousWord": int(susp_count > 0),
        "SuspiciousWordCount": susp_count,
        "IsSuspiciousTLD": int(tld in SUSPICIOUS_TLDS),
    }
    # Pastikan urutan & kelengkapan konsisten.
    return {k: feats[k] for k in FEATURE_ORDER}


def extract_registered_domain(url: str) -> str:
    """eTLD+1 sederhana untuk GROUPING split (cegah domain bocor train<->test).
    Bukan fitur model — hanya untuk membuat grup."""
    raw = normalize_url(url)
    hostname = urlparse(raw).netloc.lower().split(":")[0]
    if not hostname:
        return "unknown"
    if hostname.startswith("www."):
        hostname = hostname[4:]
    parts = [p for p in hostname.split(".") if p]
    if len(parts) <= 2:
        return hostname
    country_tlds = {"uk", "jp", "au", "nz", "br", "in", "kr", "vn", "th", "il",
                    "ca", "fr", "de", "it", "es", "ru", "cn", "tw", "hk", "sg", "id"}
    second_lvl = {"co", "com", "net", "org", "gov", "edu", "ac", "go", "ne", "or", "mil", "sch"}
    if parts[-1] in country_tlds and parts[-2] in second_lvl and len(parts) >= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


if __name__ == "__main__":
    for u in ["google.com", "https://www.google.com", "http://192.168.1.1:8080/login",
              "http://paypal.com.verify-account.secure-login.tk/webscr?cmd=update&id=1"]:
        f = extract_url_features(u)
        print(u, "->", {k: f[k] for k in list(f)[:6]}, "...")
    print("Total fitur:", len(FEATURE_ORDER))
