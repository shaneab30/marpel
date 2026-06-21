"""
phiusiil_live.py — Ekstraksi fitur ala PhiUSIIL secara LIVE (scrape) untuk model dataset.

Tujuan: menghasilkan fitur dengan NAMA KOLOM PERSIS seperti dataset PhiUSIIL
(URL + konten/HTML) supaya bisa dipakai oleh model dari skripsi_html.ipynb
(model_xgboost_dataset.pkl) untuk memprediksi URL baru secara live.

PERINGATAN (train-serving skew):
    Perhitungan fitur di sini adalah APROKSIMASI dari crawler asli PhiUSIIL.
    Tidak mungkin 100% identik (mereka pakai pipeline crawling sendiri). Jadi
    prediksi pada URL live bisa MELESET untuk sebagian situs. Ini keterbatasan
    bawaan model berbasis fitur konten dataset (lihat catatan di notebook).
    Fitur leakage (URLSimilarityIndex, TLDLegitimateProb, URLCharProb) sengaja
    TIDAK dihitung karena memang dibuang saat training.
"""

import re
import math
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

DEFAULT_TIMEOUT = 8
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
SOCIAL_DOMAINS = ("facebook.com", "twitter.com", "instagram.com", "linkedin.com",
                  "youtube.com", "tiktok.com", "t.me", "wa.me", "x.com")
_EMPTY_HREFS = {"#", "", "javascript:void(0)", "javascript:;", "javascript:"}
_IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
_PCT_RE = re.compile(r"%[0-9a-fA-F]{2}")

# Semua kolom (non-leakage) yang dihasilkan, harus mencakup FEATURES di notebook.
PHIUSIIL_FEATURES = [
    "URLLength", "DomainLength", "IsDomainIP", "CharContinuationRate", "TLDLength",
    "NoOfSubDomain", "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
    "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL", "IsHTTPS",
    "LineOfCode", "LargestLineLength", "HasTitle", "DomainTitleMatchScore",
    "URLTitleMatchScore", "HasFavicon", "Robots", "IsResponsive", "NoOfURLRedirect",
    "NoOfSelfRedirect", "HasDescription", "NoOfPopup", "NoOfiFrame",
    "HasExternalFormSubmit", "HasSocialNet", "HasSubmitButton", "HasHiddenFields",
    "HasPasswordField", "Bank", "Pay", "Crypto", "HasCopyrightInfo",
    "NoOfImage", "NoOfCSS", "NoOfJS", "NoOfSelfRef", "NoOfEmptyRef", "NoOfExternalRef",
]


def _longest_run(s, predicate):
    best = cur = 0
    for ch in s:
        if predicate(ch):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _url_part_features(url: str) -> dict:
    """Fitur dari string URL (dihitung pada URL apa adanya, termasuk skema & www
    agar definisinya mendekati dataset PhiUSIIL)."""
    parsed = urlparse(url if "://" in url else "http://" + url)
    netloc = parsed.netloc
    host = netloc.split(":")[0]
    n = max(len(url), 1)

    letters = sum(c.isalpha() for c in url)
    digits = sum(c.isdigit() for c in url)

    parts = [p for p in host.split(".") if p]
    tld = parts[-1] if parts else ""
    no_sub = max(0, len(parts) - 2)

    # Obfuscation (aproksimasi): karakter %xx ter-encode + '@'
    obf_chars = len(_PCT_RE.findall(url)) + url.count("@")

    # Karakter spesial "lain" = non-alfanumerik di luar pemisah URL standar
    allowed = set("-._~:/?#[]@!$&'()*+,;=%")
    other_special = sum(1 for c in url if (not c.isalnum()) and (c not in allowed))
    special_total = sum(1 for c in url if not c.isalnum())

    # CharContinuationRate (aproksimasi): jumlah run terpanjang huruf+digit+spesial / panjang
    lr = _longest_run(url, str.isalpha)
    dr = _longest_run(url, str.isdigit)
    sr = _longest_run(url, lambda c: not c.isalnum())
    ccr = (lr + dr + sr) / n

    return {
        "URLLength": len(url),
        "DomainLength": len(host),
        "IsDomainIP": int(bool(_IP_RE.match(host))),
        "CharContinuationRate": round(ccr, 6),
        "TLDLength": len(tld),
        "NoOfSubDomain": no_sub,
        "HasObfuscation": int(obf_chars > 0),
        "NoOfObfuscatedChar": obf_chars,
        "ObfuscationRatio": round(obf_chars / n, 6),
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": round(letters / n, 6),
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": round(digits / n, 6),
        "NoOfEqualsInURL": url.count("="),
        "NoOfQMarkInURL": url.count("?"),
        "NoOfAmpersandInURL": url.count("&"),
        "NoOfOtherSpecialCharsInURL": other_special,
        "SpacialCharRatioInURL": round(special_total / n, 6),
        "IsHTTPS": int(parsed.scheme == "https"),
    }


def _html_part_features(html: str, final_url: str, history_len: int, base_url: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    base_domain = urlparse(base_url).netloc.lower()
    low = html.lower()

    lines = html.split("\n")
    line_of_code = len(lines)
    largest_line = max((len(ln) for ln in lines), default=0)

    title_tag = soup.find("title")
    has_title = int(title_tag is not None)
    title_text = title_tag.get_text(strip=True).lower() if title_tag else ""

    dom_name = base_domain.replace("www.", "").split(".")[0]
    dom_words = set(re.split(r"[\W_]+", dom_name)) - {""}
    title_words = set(re.split(r"[\W_]+", title_text)) - {""}
    union_dt = dom_words | title_words
    dt_score = (len(dom_words & title_words) / len(union_dt) * 100.0) if union_dt else 0.0

    url_words = set(re.split(r"[\W_]+", base_url.lower())) - {""}
    union_ut = url_words | title_words
    ut_score = (len(url_words & title_words) / len(union_ut) * 100.0) if union_ut else 0.0

    images = soup.find_all("img")
    links = soup.find_all("a", href=True)
    forms = soup.find_all("form")

    empty_ref = sum(1 for a in links if a["href"].strip().lower() in _EMPTY_HREFS)
    self_ref = ext_ref = 0
    for a in links:
        href = a["href"].strip()
        if href.lower() in _EMPTY_HREFS:
            continue
        nl = urlparse(href).netloc.lower()
        if nl in ("", base_domain):
            self_ref += 1
        else:
            ext_ref += 1

    has_ext_form = 0
    for fm in forms:
        nl = urlparse((fm.get("action") or "").strip()).netloc.lower()
        if nl and nl != base_domain:
            has_ext_form = 1
            break

    return {
        "LineOfCode": line_of_code,
        "LargestLineLength": largest_line,
        "HasTitle": has_title,
        "DomainTitleMatchScore": round(dt_score, 6),
        "URLTitleMatchScore": round(ut_score, 6),
        "HasFavicon": int(soup.find("link", rel=lambda r: r and "icon" in r.lower()) is not None),
        "Robots": int(soup.find("meta", attrs={"name": re.compile("robots", re.I)}) is not None),
        "IsResponsive": int(soup.find("meta", attrs={"name": re.compile("viewport", re.I)}) is not None),
        "NoOfURLRedirect": history_len,
        "NoOfSelfRedirect": int(urlparse(final_url).netloc.lower() == base_domain and history_len > 0),
        "HasDescription": int(soup.find("meta", attrs={"name": re.compile("description", re.I)}) is not None),
        "NoOfPopup": len(soup.find_all("script", string=re.compile(r"window\.open|alert\(|popup", re.I))),
        "NoOfiFrame": len(soup.find_all("iframe")),
        "HasExternalFormSubmit": has_ext_form,
        "HasSocialNet": int(any(s in low for s in SOCIAL_DOMAINS)),
        "HasSubmitButton": int(soup.find(["input", "button"], type=lambda t: t and t.lower() == "submit") is not None),
        "HasHiddenFields": int(soup.find("input", attrs={"type": re.compile("hidden", re.I)}) is not None),
        "HasPasswordField": int(soup.find("input", attrs={"type": re.compile("password", re.I)}) is not None),
        "Bank": int(bool(re.search(r"\bbank\b", low))),
        "Pay": int(bool(re.search(r"\b(pay|payment|checkout)\b", low))),
        "Crypto": int(bool(re.search(r"\b(crypto|bitcoin|ethereum|wallet)\b", low))),
        "HasCopyrightInfo": int("©" in html or "&copy;" in low or "copyright" in low or "®" in html or "™" in html),
        "NoOfImage": len(images),
        "NoOfCSS": len(soup.find_all("link", rel="stylesheet")) + len(soup.find_all("style")),
        "NoOfJS": len(soup.find_all("script")),
        "NoOfSelfRef": self_ref,
        "NoOfEmptyRef": empty_ref,
        "NoOfExternalRef": ext_ref,
    }


_HEADERS = {
    "User-Agent": _UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
    "Connection": "keep-alive",
}


def extract_features_live(url: str, timeout: int = DEFAULT_TIMEOUT, return_reason: bool = False):
    """Scrape URL & hitung fitur PhiUSIIL (URL+HTML).
    Return dict (keys==PHIUSIIL_FEATURES), atau None bila gagal.
    Jika return_reason=True -> kembalikan (dict_atau_None, alasan_str)."""
    url_feats = _url_part_features(url)
    reason = "ok"
    try:
        resp = requests.get(url, timeout=timeout, headers=_HEADERS, allow_redirects=True)
        if resp.status_code >= 400:
            reason = f"HTTP {resp.status_code}"
            return (None, reason) if return_reason else None
        ctype = resp.headers.get("Content-Type", "").lower()
        # Tolak hanya yang JELAS bukan halaman web (json/gambar/pdf/zip).
        if any(b in ctype for b in ("application/json", "image/", "application/pdf",
                                    "application/zip", "application/octet-stream")):
            reason = f"bukan HTML ({ctype})"
            return (None, reason) if return_reason else None
        html_feats = _html_part_features(resp.text, resp.url, len(resp.history), url)
    except requests.exceptions.Timeout:
        reason = "timeout"
        return (None, reason) if return_reason else None
    except requests.exceptions.SSLError:
        reason = "ssl_error"
        return (None, reason) if return_reason else None
    except requests.exceptions.ConnectionError:
        reason = "connection_error (situs mati/diblokir)"
        return (None, reason) if return_reason else None
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
        return (None, reason) if return_reason else None

    feats = {**url_feats, **html_feats}
    out = {k: feats.get(k, 0) for k in PHIUSIIL_FEATURES}
    return (out, reason) if return_reason else out


def _fetch_html_requests(url, timeout):
    """Ambil HTML via requests (cepat, tapi tak menjalankan JS). Return (html, final_url) / None."""
    try:
        resp = requests.get(url, timeout=timeout, headers=_HEADERS, allow_redirects=True)
        if resp.status_code >= 400:
            return None
        ctype = resp.headers.get("Content-Type", "").lower()
        if any(b in ctype for b in ("application/json", "image/", "application/pdf",
                                    "application/zip", "application/octet-stream")):
            return None
        return resp.text, resp.url
    except Exception:
        return None


def _fetch_html_browser(url, timeout):
    """Render halaman dengan headless Chromium (Playwright) -> (html, final_url) / None.
    Lebih tahan situs JS / anti-bot (mis. Cloudflare) dibanding requests.
    Perlu: pip install playwright  &&  playwright install chromium.
    Aman di backend Flask (thread tanpa event loop). Di Jupyter (ada asyncio loop)
    sync API akan error -> ditangkap -> None -> fallback ke requests."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(user_agent=_UA, locale="en-US")
            page = ctx.new_page()
            page.goto(url, timeout=timeout * 1000, wait_until="domcontentloaded")
            try:
                page.wait_for_load_state("networkidle", timeout=timeout * 1000)
            except Exception:
                pass
            html, final_url = page.content(), page.url
            browser.close()
            return html, final_url
    except Exception:
        return None


def scrape_html_features(url: str, timeout: int = DEFAULT_TIMEOUT, render: bool = False):
    """Scrape & kembalikan HANYA fitur HTML (nama kolom dataset), atau None bila gagal.
    render=True -> coba headless browser (Playwright) dulu untuk situs JS/anti-bot,
                   lalu fallback ke requests bila gagal/tak tersedia.
    Dipakai backend jalur FULL: fitur URL dihitung TERPISAH via url_features.py."""
    fetched = None
    if render:
        fetched = _fetch_html_browser(url, timeout)
    if fetched is None:
        fetched = _fetch_html_requests(url, timeout)
    if fetched is None:
        return None
    html, final_url = fetched
    try:
        # history_len=0: fitur redirect termasuk fitur 'rapuh' & tak dipakai model FULL.
        return _html_part_features(html, final_url, 0, url)
    except Exception:
        return None


if __name__ == "__main__":
    for u in ["https://www.google.com", "https://www.youtube.com", "https://github.com"]:
        f, why = extract_features_live(u, return_reason=True)
        print(u, "->", "OK" if f else f"GAGAL ({why})")