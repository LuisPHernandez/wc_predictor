from bs4 import BeautifulSoup # pyrefly: ignore [missing-import]
import pandas as pd
import numpy as np
import re
import time
from selenium import webdriver # pyrefly: ignore [missing-import]
from selenium.webdriver.common.by import By # pyrefly: ignore [missing-import]
from selenium.webdriver.chrome.service import Service # pyrefly: ignore [missing-import]
from webdriver_manager.chrome import ChromeDriverManager # pyrefly: ignore [missing-import]
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------------- Config ------------------------------------------

URLS = [
    "https://www.cuotasahora.com/football/world/fase-final-copa-del-mundo-2014/results/",
    "https://www.cuotasahora.com/football/world/fase-final-copa-del-mundo-2018/results/",
    "https://www.cuotasahora.com/football/world/fase-final-copa-del-mundo-2022/results/",
]
OUTPUT_FILE = PROJECT_ROOT / "scraper" / "wc_btts.csv"

# ----------------------------------- Helpers ------------------------------------------

def collect_match_links(driver, url):
    driver.get("about:blank")
    time.sleep(1)
    driver.get(url)
    time.sleep(6)

    html = BeautifulSoup(driver.page_source, "html.parser")
    matches = []
    for row in html.select('[data-testid="game-row"]'):
        teams = [x.get_text(strip=True) for x in row.select(".participant-name")]
        if len(teams) != 2:
            continue
        link_el = row.select_one("a[href]")
        if not link_el:
            continue
        matches.append((teams[0], teams[1], link_el["href"]))
    return matches


def get_btts_data(driver, match_href):
    # Land on 1x2 page, then click O/U tab
    driver.get("about:blank")
    time.sleep(1)
    driver.get("https://www.cuotasahora.com" + match_href)
    time.sleep(8)

    try:
        btts_tab = driver.find_element(By.XPATH, '//a[.//div[text()="Ambos equipos marcan"]]')
        driver.execute_script('arguments[0].click();', btts_tab)
        time.sleep(8)
    except Exception as e:
        print(f"Could not click btts tab: {e}")
        return None, None, None

    html2 = BeautifulSoup(driver.page_source, "html.parser")
    rows = html2.select('[data-testid="over-under-expanded-row"]')

    yes_odds  = []
    no_odds = []

    for row in rows:
        odds_els = row.select("p.odds-text")
        if len(odds_els) < 2:
            continue
        try:
            yes_odds.append(float(odds_els[0].get_text(strip=True)))
            no_odds.append(float(odds_els[1].get_text(strip=True)))
        except ValueError:
            continue

    if not yes_odds:
        print(f"Odds not available for {match_href}, continuing...")
        return

    avg_yes  = round(np.mean(yes_odds),  4)
    avg_no = round(np.mean(no_odds), 4)

# ----------------------------------- Main ------------------------------------------

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Resume from existing CSV — skip matches that already have odds
try:
    existing = pd.read_csv(OUTPUT_FILE)
    # Only consider rows that already have both odds columns populated
    complete = existing.dropna(subset=["avg_yes", "avg_no"])
    seen = set(zip(complete["home_team"], complete["away_team"]))
    dataset = existing.to_dict("records")
    print(f"Resuming — {len(seen)} matches already have odds, {len(existing) - len(seen)} incomplete.")
except FileNotFoundError:
    seen = set()
    dataset = []
    print("No existing CSV found, starting fresh.")

for results_url in URLS:
    print(f"\n--- {results_url} ---")

    all_matches = []
    for page in [1, 2]:
        url = results_url if page == 1 else f"{results_url}#/page/2/"
        if page == 2:
            driver.get("about:blank")
            time.sleep(1)
            driver.get(results_url)
            time.sleep(6)
        matches = collect_match_links(driver, url)
        print(f"Page {page}: found {len(matches)} matches")
        all_matches.extend(matches)
        if len(matches) == 0:
            break

    print(f"Total: {len(all_matches)} matches to scrape")

    for i, (home, away, href) in enumerate(all_matches):
        if (home, away) in seen:
            print(f"  [{i+1}/{len(all_matches)}] {home} vs {away} — already scraped, skipping")
            continue

        try:
            avg_yes, avg_no = get_btts_data(driver, href)

            if avg_yes is None:
                print(f"  [{i+1}/{len(all_matches)}] {home} vs {away} — no btts data, skipping")
                continue

            # Update existing row if present, else append
            match_found = False
            for rec in dataset:
                if rec["home_team"] == home and rec["away_team"] == away:
                    rec["avg_yes"]  = avg_yes
                    rec["avg_no"] = avg_no
                    match_found = True
                    break
            if not match_found:
                dataset.append({
                    "home_team": home,
                    "away_team": away,
                    "avg_over":  avg_yes,
                    "avg_under": avg_no,
                })
            seen.add((home, away))
            pd.DataFrame(dataset).to_csv(OUTPUT_FILE, index=False)
            print(f"  [{i+1}/{len(all_matches)}] {home} vs {away} → yes={avg_yes} no={avg_no}")

        except Exception as e:
            print(f"  [{i+1}/{len(all_matches)}] ERROR {home} vs {away}: {e}")

driver.quit()

df = pd.DataFrame(dataset)
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nDone — {len(df)} matches saved to {OUTPUT_FILE}")
print(df.to_string())