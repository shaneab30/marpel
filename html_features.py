"""
html_features.py — Scrape + ekstraksi fitur KONTEN/HTML dari sebuah URL.

PENTING (anti train-serving skew):
    Modul ini SATU-SATUNYA sumber kebenaran untuk fitur HTML.
    - Notebook (skripsi.ipynb) memakai fungsi ini untuk men-SCRAPE ULANG URL dataset
      saat training -> fitur HTML training dihitung dengan kode yang SAMA dengan backend.
    - Backend (app.py) memakai fungsi ini saat inference.
    Karena scraper & definisi fiturnya identik, nilai fitur HTML saat training == produksi.

Strategi anti-leakage:
    - Hanya halaman yang BERHASIL di-scrape yang dipakai. Halaman gagal/mati TIDAK
      diisi 0 lalu dilatih (itu menimbulkan aturan palsu "halaman kosong = phishing").
      Di training: baris gagal scrape dibuang. Di backend: jika scrape gagal -> pakai
      model URL-only (fallback), bukan mengisi 0.
"""

import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

DEFAULT_TIMEOUT = 6
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

SOCIAL_DOMAINS = ("facebook.com", "twitter.com", "instagram.com", "linkedin.com",
                  "youtube.com", "tiktok.com", "t.me", "wa.me", "x.com")

# Urutan kolom fitur HTML. WAJIB identik antara training & inference.
HTML_FEATURE_ORDER = [
    "LineOfCode", "LargestLineLength",
    "HasTitle", "DomainTitleMatchScore", "HasDescription", "HasFavicon",
    "HasCopyrightInfo", "HasSubmitButton",
    "NoOfImage", "NoOfCSS", "NoOfJS",
    "NoOfSelfRef", "NoOfEmptyRef", "NoOfExternalRef",
    "NoOfPopup", "NoOfiFrame",
    "NoOfForms", "HasPasswordField", "HasHiddenField", "HasExternalFormAction",
    "HasSocialNet", "Pay",
]

_EMPTY_HREFS = {"#", "", "javascript:void(0)", "javascript:;", "javascript:"}


def fetch_html(url: str, timeout: int = DEFAULT_TIMEOUT):
    """Ambil HTML mentah. Return (html, final_url) atau None bila gagal/mati."""
    try:
        resp = requests.get(url, timeout=timeout,
                            headers={"User-Agent": _UA},
                            allow_redirects=True)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if "html" not in ctype.lower() and ctype:
            return None
        return resp.text, resp.url
    except Exception:
        return None


def extract_html_features(html: str, url: str) -> dict:
    """Hitung fitur dari HTML yang sudah diambil. Key-nya == HTML_FEATURE_ORDER."""
    soup = BeautifulSoup(html, "html.parser")
    base_domain = urlparse(url).netloc.lower()

    lines = html.split("\n")
    line_of_code = len(lines)
    largest_line = max((len(ln) for ln in lines), default=0)

    title_tag = soup.find("title")
    has_title = int(title_tag is not None)
    title_text = title_tag.get_text(strip=True).lower() if title_tag else ""

    dom_name = base_domain.replace("www.", "").split(".")[0]
    dom_words = set(re.split(r"[\W_]+", dom_name)) - {""}
    title_words = set(re.split(r"[\W_]+", title_text)) - {""}
    union = dom_words | title_words
    dt_score = (len(dom_words & title_words) / len(union) * 100.0) if union else 0.0

    has_desc = int(soup.find("meta", attrs={"name": re.compile("description", re.I)}) is not None)
    low_html = html.lower()
    has_copyright = int("©" in html or "&copy;" in low_html or "copyright" in low_html
                        or "™" in html or "®" in html)
    has_social = int(any(s in low_html for s in SOCIAL_DOMAINS))

    images = soup.find_all("img")
    links = soup.find_all("a", href=True)
    forms = soup.find_all("form")

    has_favicon = int(soup.find("link", rel=lambda r: r and "icon" in r.lower()) is not None)
    has_submit = int(soup.find(["input", "button"], type=lambda t: t and t.lower() == "submit") is not None)
    pay = int(bool(re.search(r"\b(pay|payment|checkout|credit\s*card|cvv)\b", low_html)))
    no_css = len(soup.find_all("link", rel="stylesheet")) + len(soup.find_all("style"))
    no_js = len(soup.find_all("script"))

    empty_ref = sum(1 for a in links if a["href"].strip().lower() in _EMPTY_HREFS)
    self_ref, ext_ref = 0, 0
    for a in links:
        href = a["href"].strip()
        if href.lower() in _EMPTY_HREFS:
            continue
        netloc = urlparse(href).netloc.lower()
        if netloc in ("", base_domain):
            self_ref += 1
        else:
            ext_ref += 1

    no_popup = len(soup.find_all("script", string=re.compile(r"window\.open|alert\(|popup", re.I)))
    no_iframe = len(soup.find_all("iframe"))

    has_password = int(soup.find("input", attrs={"type": re.compile("password", re.I)}) is not None)
    has_hidden = int(soup.find("input", attrs={"type": re.compile("hidden", re.I)}) is not None)

    has_ext_form = 0
    for fm in forms:
        action = (fm.get("action") or "").strip()
        netloc = urlparse(action).netloc.lower()
        if netloc and netloc != base_domain:
            has_ext_form = 1
            break

    feats = {
        "LineOfCode": line_of_code,
        "LargestLineLength": largest_line,
        "HasTitle": has_title,
        "DomainTitleMatchScore": round(dt_score, 6),
        "HasDescription": has_desc,
        "HasFavicon": has_favicon,
        "HasCopyrightInfo": has_copyright,
        "HasSubmitButton": has_submit,
        "NoOfImage": len(images),
        "NoOfCSS": no_css,
        "NoOfJS": no_js,
        "NoOfSelfRef": self_ref,
        "NoOfEmptyRef": empty_ref,
        "NoOfExternalRef": ext_ref,
        "NoOfPopup": no_popup,
        "NoOfiFrame": no_iframe,
        "NoOfForms": len(forms),
        "HasPasswordField": has_password,
        "HasHiddenField": has_hidden,
        "HasExternalFormAction": has_ext_form,
        "HasSocialNet": has_social,
        "Pay": pay,
    }
    return {k: feats[k] for k in HTML_FEATURE_ORDER}


def scrape_and_extract(url: str, timeout: int = DEFAULT_TIMEOUT):
    """Convenience: fetch + extract. Return dict fitur HTML atau None bila gagal."""
    res = fetch_html(url, timeout=timeout)
    if res is None:
        return None
    html, final_url = res
    try:
        return extract_html_features(html, final_url)
    except Exception:
        return None


if __name__ == "__main__":
    for u in ["https://www.google.com", "https://github.com"]:
        f = scrape_and_extract(u)
        print(u, "->", "GAGAL" if f is None else {k: f[k] for k in list(HTML_FEATURE_ORDER)[:6]})
