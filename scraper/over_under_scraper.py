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
    "https://www.cuotasahora.com/football/world/campeonato-del-mundo-2026/",
]
OUTPUT_FILE = PROJECT_ROOT / "scraper" / "wc_expected_goals.csv"

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


def get_ou_data(driver, match_href):
    # Land on 1x2 page, then click O/U tab
    driver.get("about:blank")
    time.sleep(1)
    driver.get("https://www.cuotasahora.com" + match_href)
    time.sleep(8)

    try:
        ou_tab = driver.find_element(By.XPATH, '//a[.//div[text()="Más/Menos de"]]')
        driver.execute_script('arguments[0].click();', ou_tab)
        time.sleep(8)
    except Exception as e:
        print(f"    Could not click O/U tab: {e}")
        return None, None, None

    # Step 1: find best collapsed row (smallest |over - under| from summary odds)
    html = BeautifulSoup(driver.page_source, "html.parser")
    collapsed_rows = html.select('[data-testid="over-under-collapsed-row"]')

    best_line = None
    best_diff = float("inf")
    best_idx = None

    for idx, row in enumerate(collapsed_rows):
        option_box = row.select_one('[data-testid="over-under-collapsed-option-box"]')
        if not option_box:
            continue
        label_text = option_box.find("p").get_text(strip=True) if option_box.find("p") else ""
        m = re.search(r"\+?([\d.]+)", label_text)
        if not m:
            continue
        goal_line = float(m.group(1))

        odds_els = row.select('p[data-testid="odd-container-default"]')
        if len(odds_els) < 2:
            continue
        try:
            over_odd  = float(odds_els[0].get_text(strip=True))
            under_odd = float(odds_els[1].get_text(strip=True))
        except ValueError:
            continue

        diff = abs(over_odd - under_odd)
        if diff < best_diff:
            best_diff = diff
            best_line = goal_line
            best_idx  = idx

    if best_idx is None:
        return None, None, None

    # Step 2: click the best collapsed row to expand it
    live_rows = driver.find_elements(By.CSS_SELECTOR, '[data-testid="over-under-collapsed-row"]')
    if best_idx >= len(live_rows):
        return None, None, None

    driver.execute_script('arguments[0].click();', live_rows[best_idx])
    time.sleep(4)

    # Step 3: scrape per-bookmaker odds from expanded rows
    html2 = BeautifulSoup(driver.page_source, "html.parser")
    expanded_rows = html2.select('[data-testid="over-under-expanded-row"]')

    over_odds  = []
    under_odds = []

    for row in expanded_rows:
        odds_els = row.select("p.odds-text")
        if len(odds_els) < 2:
            continue
        try:
            over_odds.append(float(odds_els[0].get_text(strip=True)))
            under_odds.append(float(odds_els[1].get_text(strip=True)))
        except ValueError:
            continue

    if not over_odds:
        return best_line, None, None

    avg_over  = round(np.mean(over_odds),  4)
    avg_under = round(np.mean(under_odds), 4)

    return best_line, avg_over, avg_under

# ----------------------------------- Main ------------------------------------------

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Resume from existing CSV — skip matches that already have odds
try:
    existing = pd.read_csv(OUTPUT_FILE)
    # Only consider rows that already have both odds columns populated
    complete = existing.dropna(subset=["avg_over", "avg_under"])
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
        print(f"  Page {page}: found {len(matches)} matches")
        all_matches.extend(matches)
        if len(matches) == 0:
            break

    print(f"  Total: {len(all_matches)} matches to scrape")

    for i, (home, away, href) in enumerate(all_matches):
        if (home, away) in seen:
            print(f"  [{i+1}/{len(all_matches)}] {home} vs {away} — already scraped, skipping")
            continue

        try:
            ou_line, avg_over, avg_under = get_ou_data(driver, href)

            if ou_line is None:
                print(f"  [{i+1}/{len(all_matches)}] {home} vs {away} — no O/U data, skipping")
                continue

            # Update existing row if present, else append
            match_found = False
            for rec in dataset:
                if rec["home_team"] == home and rec["away_team"] == away:
                    rec["ou_line"]   = ou_line
                    rec["avg_over"]  = avg_over
                    rec["avg_under"] = avg_under
                    match_found = True
                    break
            if not match_found:
                dataset.append({
                    "home_team": home,
                    "away_team": away,
                    "ou_line":   ou_line,
                    "avg_over":  avg_over,
                    "avg_under": avg_under,
                })
            seen.add((home, away))
            pd.DataFrame(dataset).to_csv(OUTPUT_FILE, index=False)
            print(f"  [{i+1}/{len(all_matches)}] {home} vs {away} → line={ou_line} over={avg_over} under={avg_under}")

        except Exception as e:
            print(f"  [{i+1}/{len(all_matches)}] ERROR {home} vs {away}: {e}")

driver.quit()

df = pd.DataFrame(dataset)
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nDone — {len(df)} matches saved to {OUTPUT_FILE}")
print(df.to_string())