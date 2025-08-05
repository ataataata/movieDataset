#!/usr/bin/env python3
"""

 Libaries needed: selenium webdriver-manager beautifulsoup4 pandas

How to use:
    python script.py "https://clip.cafe/the-wolf-of-wall-street-2013/the-qualude/"
    link is from clip.cafe it can be any clip, there are no daily download restrictions!

Output:
    • data.csv        updated with a new row that includes: id, actor, movie, line, duration
    • Lines/xx.wav    the audio clip (xx = id)
"""

import os, re, time, pickle, sys, csv, tempfile, shutil, pathlib
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

CSV_PATH        = "data.csv"
DL_DIR          = pathlib.Path("Lines")
COOKIES_FILE    = "clipcafe_cookies.pkl"
WAIT_SECS       = 15
LOGIN_WAIT_SECS = 60 

def start_browser():
    opts = webdriver.ChromeOptions()
    opts.add_argument("--start-maximized")

    tmp = tempfile.mkdtemp()
    prefs = {"download.default_directory": tmp,
             "download.prompt_for_download": False}
    opts.add_experimental_option("prefs", prefs)

    service = Service(ChromeDriverManager().install())
    driver  = webdriver.Chrome(service=service, options=opts)
    driver.tmp_download_dir = pathlib.Path(tmp)
    return driver

def restore_cookies(driver):
    if not os.path.exists(COOKIES_FILE):
        return False
    driver.get("https://clip.cafe")
    with open(COOKIES_FILE, "rb") as f:
        for c in pickle.load(f):
            driver.add_cookie(c)
    return True

def save_cookies(driver):
    with open(COOKIES_FILE, "wb") as f:
        pickle.dump(driver.get_cookies(), f)

def next_id():
    if not os.path.exists(CSV_PATH):
        return 1
    return int(pd.read_csv(CSV_PATH)["id"].max()) + 1

def append_csv(row):
    hdr = ["id", "Actor Name", "Movie Name", "Line", "Duration"]
    need_header = not os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        if need_header:
            w.writeheader()
        w.writerow(row)

def extract_metadata(html: str):
    soup = BeautifulSoup(html, "html.parser")

    cast = soup.select_one("div.movieCastActor")
    if cast:
        character = cast.b.get_text(strip=True)
        actor     = cast.a.get_text(strip=True)
    else:
        character = soup.select_one("div.highlight-box b").get_text(strip=True)
        actor     = character

    movie = soup.select_one("a.white.pl-10").get_text(strip=True)

    quote_div = soup.select_one("div.highlight-box")
    full_text = quote_div.get_text(" ", strip=True)

    line = re.sub(rf"^{re.escape(character)}\s*:\s*", "", full_text, flags=re.I)

    dur_match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*secs?", html, re.I)
    duration  = float(dur_match.group(1)) if dur_match else 0.0

    return actor, movie, line, duration

def wait_for_download(tmp_dir: pathlib.Path, before):
    deadline = time.time() + WAIT_SECS
    while time.time() < deadline:
        files = list(tmp_dir.glob("*.wav"))
        new   = [f for f in files if f not in before]
        if new:
            return new[0]
        time.sleep(1)
    raise RuntimeError("Timed out waiting for WAV download.")

def process_clip(url: str):
    drv = start_browser()
    try:
        first_run = not restore_cookies(drv)

        if first_run:
            print(f"🔑 First run – log in within {LOGIN_WAIT_SECS}s…")
            drv.get("https://clip.cafe")
            for sec in range(LOGIN_WAIT_SECS, 0, -1):
                print(f"\r   {sec:02d}s remaining…", end="", flush=True)
                time.sleep(1)
            print("\n💾 Saving cookies.")
            save_cookies(drv)

        drv.get(url)

        WebDriverWait(drv, WAIT_SECS).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.highlight-box")))
        actor, movie, line, duration = extract_metadata(drv.page_source)

        WebDriverWait(drv, WAIT_SECS).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR,
                "button[aria-label='Download Clip']"))).click()
        WebDriverWait(drv, WAIT_SECS).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "input#audio-wav"))).click()
        drv.find_element(By.CSS_SELECTOR,
            "button.orangeButton[type='submit']").click()

        before   = set(drv.tmp_download_dir.glob("*.wav"))
        wav_path = wait_for_download(drv.tmp_download_dir, before)

        clip_id   = next_id()
        DL_DIR.mkdir(exist_ok=True)
        final_wav = DL_DIR / f"{clip_id:02d}.wav"
        shutil.move(str(wav_path), final_wav)

        append_csv({
            "id": clip_id,
            "Actor Name": actor,
            "Movie Name": movie,
            "Line": line,
            "Duration": duration
        })

        print(f"\n✅ Added id {clip_id:02d}: {actor} — “{movie}”")
        print(f"   ↳ Audio saved to {final_wav}")

    finally:
        drv.quit()
        shutil.rmtree(drv.tmp_download_dir, ignore_errors=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <clip_url>")
        sys.exit(1)
    process_clip(sys.argv[1])