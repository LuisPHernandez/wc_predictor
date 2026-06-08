from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ----------------------------------- Config ------------------------------------------

URLS = [
    "https://www.cuotasahora.com/football/world/copa-del-mundo-2022/results/",
    "https://www.cuotasahora.com/football/world/copa-del-mundo-2018/results/",
    "https://www.cuotasahora.com/football/world/copa-del-mundo-2014/results/",
]
OUTPUT_FILE = "wc_expected_goals.csv"

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


def get_ou_line(driver, match_href):
    # Land on 1x2 page first (full page load)
    driver.get("about:blank")
    time.sleep(1)
    driver.get("https://www.cuotasahora.com" + match_href)
    time.sleep(8)

    # Click the O/U tab via JS — direct URL navigation gets redirected back to 1x2
    try:
        ou_tab = driver.find_element(By.XPATH, '//a[.//div[text()="Más/Menos de"]]')
        driver.execute_script('arguments[0].click();', ou_tab)
        time.sleep(8)
    except Exception as e:
        print(f"    Could not click O/U tab: {e}")
        return None

    html = BeautifulSoup(driver.page_source, "html.parser")
    rows = html.select('[data-testid="over-under-collapsed-row"]')

    best_line = None
    best_diff = float("inf")

    for row in rows:
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

    return best_line

# ----------------------------------- Main ------------------------------------------

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

dataset = []

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
        try:
            ou_line = get_ou_line(driver, href)

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