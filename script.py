#!/usr/bin/env python3
"""
Clip.Cafe batch scraper–downloader

Libraries:
  pip install selenium webdriver-manager beautifulsoup4 pandas bs4

USAGE
• One URL:
      python script.py "https://clip.cafe/the-wolf-of-wall-street-2013/the-qualude/"
• Many URLs:
      python script.py <url1> <url2> <url3>
• Text file (one URL per line):
      python script.py links.txt

Outputs
• data.csv         – grows with every clip (safe quoting)
• Lines/NN.wav     – WAVs named after their id (01.wav, 02.wav, …)
"""

import os, re, sys, csv, time, pickle, tempfile, shutil, pathlib
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ─── paths & settings ───────────────────────────────────────────
CSV_PATH        = "data.csv"
DL_DIR          = pathlib.Path("Lines")
COOKIES_FILE    = "clipcafe_cookies.pkl"
WAIT_SECS       = 15          # generic Selenium wait
LOGIN_WAIT_SECS = 60          # first-run login grace period

# ─── browser helpers ───────────────────────────────────────────
def start_browser():
    opts = webdriver.ChromeOptions()
    opts.add_argument("--start-maximized")
    # temp dir so we can detect the new WAV reliably
    tmp = tempfile.mkdtemp()
    opts.add_experimental_option("prefs", {
        "download.default_directory": tmp,
        "download.prompt_for_download": False,
    })
    drv = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                           options=opts)
    drv.tmp_download_dir = pathlib.Path(tmp)
    return drv

def restore_cookies(driver):
    if not os.path.exists(COOKIES_FILE):
        return False
    driver.get("https://clip.cafe")
    with open(COOKIES_FILE, "rb") as fh:
        for c in pickle.load(fh):
            driver.add_cookie(c)
    return True

def save_cookies(driver):
    with open(COOKIES_FILE, "wb") as fh:
        pickle.dump(driver.get_cookies(), fh)

# ─── CSV helpers ────────────────────────────────────────────────
def _safe_max_id_from_csv(path: str) -> int | None:
    """Read only the id column, skipping malformed lines. Return max id or None."""
    try:
        df = pd.read_csv(
            path,
            engine="python",         # needed for on_bad_lines="skip"
            on_bad_lines="skip",
            usecols=["id"],
            dtype={"id": "Int64"},
        )
        if df.empty or df["id"].dropna().empty:
            return None
        return int(df["id"].dropna().astype(int).max())
    except Exception:
        # csv fallback parser (skip bad rows silently)
        try:
            max_id = None
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    try:
                        val = int(row.get("id", "").strip())
                        max_id = val if (max_id is None or val > max_id) else max_id
                    except Exception:
                        continue
            return max_id
        except Exception:
            return None

def _safe_max_id_from_files(folder: pathlib.Path) -> int | None:
    """Look for Lines/NN.wav and return max NN or None."""
    if not folder.exists():
        return None
    nums = []
    for p in folder.glob("*.wav"):
        try:
            nums.append(int(p.stem))
        except Exception:
            continue
    return max(nums) if nums else None

def next_id() -> int:
    """Robust next id from CSV first, then from existing WAV filenames, else 1."""
    if not os.path.exists(CSV_PATH):
        file_max = _safe_max_id_from_files(DL_DIR)
        return (file_max + 1) if file_max is not None else 1

    csv_max = _safe_max_id_from_csv(CSV_PATH)
    file_max = _safe_max_id_from_files(DL_DIR)

    candidates = [x for x in [csv_max, file_max] if x is not None]
    return (max(candidates) + 1) if candidates else 1

def append_csv(row: dict):
    header = ["id", "Actor Name", "Movie Name", "Line", "Duration"]
    write_header = not os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=header,
            quoting=csv.QUOTE_MINIMAL,   # quote fields that need it
            escapechar="\\",
        )
        if write_header:
            w.writeheader()
        w.writerow(row)

# ─── scraping helpers ──────────────────────────────────────────
def extract_meta(html: str):
    """
    Return (actor, movie, cleaned_line, duration), handling pages that
    lack the grey “Character by Actor” bar or any actor link at all.
    """
    soup = BeautifulSoup(html, "html.parser")

    cast = soup.select_one("div.movieCastActor")
    if cast and cast.select_one("a[href*='actor/']"):
        char  = cast.select_one("b").get_text(strip=True)
        actor = cast.select_one("a[href*='actor/']").get_text(strip=True)
    else:
        a_tag = soup.select_one("a[href*='actor/']")
        actor = a_tag.get_text(strip=True) if a_tag else "Unknown"
        char_tag = soup.select_one("div.highlight-box b")
        char  = char_tag.get_text(strip=True) if char_tag else ""

    movie = soup.select_one("a.white.pl-10").get_text(strip=True)

    quote_box = soup.select_one("div.highlight-box")
    quote_txt = quote_box.get_text(" ", strip=True)
    line = (re.sub(rf"^{re.escape(char)}\s*:\s*", "", quote_txt, flags=re.I)
            if char else quote_txt)

    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*secs?", html, re.I)
    duration = float(m.group(1)) if m else 0.0
    return actor, movie, line, duration

def wait_download(tmp_dir: pathlib.Path, before):
    deadline = time.time() + WAIT_SECS
    while time.time() < deadline:
        # ignore partial Chrome files (.crdownload)
        files = [f for f in tmp_dir.glob("*.wav")]
        new   = [f for f in files if f not in before]
        if new:
            return new[0]
        time.sleep(1)
    raise RuntimeError("Download timed out")

# ─── process one clip (driver already running) ─────────────────
def handle_clip(drv, url):
    drv.get(url)
    WebDriverWait(drv, WAIT_SECS).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.highlight-box")))

    actor, movie, line, dur = extract_meta(drv.page_source)

    # snapshot BEFORE clicking to avoid race with fast downloads
    before = set(drv.tmp_download_dir.glob("*.wav"))

    # overlay-proof click on “Download Clip”
    drv.execute_script("document.querySelectorAll('.fixedBanner').forEach(el=>el.remove())")
    btn = WebDriverWait(drv, WAIT_SECS).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label='Download Clip']")))
    drv.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
    drv.execute_script("arguments[0].click();", btn)

    WebDriverWait(drv, WAIT_SECS).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "input#audio-wav"))).click()
    drv.find_element(By.CSS_SELECTOR, "button.orangeButton[type='submit']").click()

    wav = wait_download(drv.tmp_download_dir, before)

    idx = next_id()
    DL_DIR.mkdir(exist_ok=True)
    shutil.move(str(wav), DL_DIR / f"{idx:02d}.wav")

    append_csv({"id": idx, "Actor Name": actor, "Movie Name": movie,
                "Line": line, "Duration": dur})
    print(f"✔ {idx:02d}  {actor} — “{movie}”")

# ─── run a batch of links ──────────────────────────────────────
def run_batch(urls):
    drv = start_browser()
    try:
        if not restore_cookies(drv):
            print(f"First run – log in within {LOGIN_WAIT_SECS}s …")
            drv.get("https://clip.cafe")
            for s in range(LOGIN_WAIT_SECS, 0, -1):
                print(f"\r   {s:02d}s left", end="", flush=True)
                time.sleep(1)
            print("\nCookies saved.")
            save_cookies(drv)

        for u in urls:
            try:
                handle_clip(drv, u)
            except Exception as e:
                print(f"error: {u}  ({e})")

    finally:
        drv.quit()
        shutil.rmtree(drv.tmp_download_dir, ignore_errors=True)

# ─── utilities ─────────────────────────────────────────────────
def collect_urls(args):
    if len(args) == 1 and os.path.isfile(args[0]):         # links.txt style
        with open(args[0], encoding="utf-8") as fh:
            return [ln.strip() for ln in fh if ln.strip()]
    return args

# ─── main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:\n  python script.py <URL …>\n  python script.py links.txt")
        sys.exit(1)
    run_batch(collect_urls(sys.argv[1:]))
