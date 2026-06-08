from bs4 import BeautifulSoup # pyrefly: ignore [missing-import]
import pandas as pd
import re
import time
from selenium import webdriver # pyrefly: ignore [missing-import]
from selenium.webdriver.chrome.service import Service # pyrefly: ignore [missing-import]
from webdriver_manager.chrome import ChromeDriverManager # pyrefly: ignore [missing-import]

# ----------------------------------- Config ------------------------------------------

URLS = [
    "https://www.cuotasahora.com/football/world/copa-del-mundo-2022/results/",
    "https://www.cuotasahora.com/football/world/copa-del-mundo-2018/results/",
    "https://www.cuotasahora.com/football/world/copa-del-mundo-2014/results/",
]
OUTPUT_FILE = "wc_expected_goals.csv"

# ----------------------------------- Helpers ------------------------------------------

def collect_match_links(driver, url):
    """Navigate to results page and collect all (home, away, ou_url) tuples."""
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
        ou_url = "https://www.cuotasahora.com" + link_el["href"].rstrip("/") + "/:over-under;2"
        matches.append((teams[0], teams[1], ou_url))

    return matches


def get_ou_line(driver, ou_url):
    """
    Navigate to O/U page and find the goal line where over/under odds are closest.
    Parses goal line from the option box text, odds from odd-container-default p tags.
    """
    driver.get(ou_url)
    time.sleep(8)

    html = BeautifulSoup(driver.page_source, "html.parser")
    rows = html.select('[data-testid="over-under-collapsed-row"]')

    best_line = None
    best_diff = float("inf")

    for row in rows:
        # Goal line: grab the full text of the option box and regex out the number
        option_box = row.select_one('[data-testid="over-under-collapsed-option-box"]')
        if not option_box:
            continue
        # Use the first <p> tag text (avoids mobile duplicate)
        label_text = option_box.find("p").get_text(strip=True) if option_box.find("p") else ""
        m = re.search(r"\+?([\d.]+)", label_text)
        if not m:
            continue
        goal_line = float(m.group(1))

        # Odds: all p tags with data-testid="odd-container-default" inside this row
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

    return best_line

# ----------------------------------- Main ------------------------------------------

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

dataset = []

for results_url in URLS:
    print(f"\n--- {results_url} ---")

    all_matches = []
    for page in [1, 2]:
        url = results_url if page == 1 else f"{results_url}#/page/2/"
        matches = collect_match_links(driver, url)
        print(f"  Page {page}: found {len(matches)} matches")
        all_matches.extend(matches)
        if len(matches) == 0:
            break  # no second page

    print(f"  Total: {len(all_matches)} matches to scrape")

    for i, (home, away, ou_url) in enumerate(all_matches):
        try:
            ou_line = get_ou_line(driver, ou_url)

            if ou_line is None:
                print(f"  [{i+1}/{len(all_matches)}] {home} vs {away} — no O/U data, skipping")
                continue

            dataset.append({"home_team": home, "away_team": away, "ou_line": ou_line})
            pd.DataFrame(dataset).to_csv(OUTPUT_FILE, index=False)
            print(f"  [{i+1}/{len(all_matches)}] {home} vs {away} → {ou_line}")

        except Exception as e:
            print(f"  [{i+1}/{len(all_matches)}] ERROR {home} vs {away}: {e}")

driver.quit()

df = pd.DataFrame(dataset)
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nDone — {len(df)} matches saved to {OUTPUT_FILE}")
print(df.to_string())