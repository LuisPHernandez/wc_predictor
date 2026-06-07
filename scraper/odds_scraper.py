from bs4 import BeautifulSoup # pyrefly: ignore [missing-import]
import pandas as pd
import numpy as np
import time
from selenium import webdriver # pyrefly: ignore [missing-import]
from selenium.webdriver.chrome.service import Service # pyrefly: ignore [missing-import]
from webdriver_manager.chrome import ChromeDriverManager # pyrefly: ignore [missing-import]

# ----------------------------------- Helpers ------------------------------------------

month_map = {
    "ene.": "01",
    "feb.": "02",
    "mar.": "03",
    "abr.": "04",
    "may.": "05",
    "jun.": "06",
    "jul.": "07",
    "ago.": "08",
    "agto.": "08",
    "sep.": "09",
    "oct.": "10",
    "nov.": "11",
    "dic.": "12",
}

def parse_spanish_date(date_text):
    day, month, year = date_text.split()
    return f"{year}-{month_map[month]}-{int(day):02d}"

#  ----------------------------------- Scraper ------------------------------------------

# URLs to scrape

urls = [
    # Copa África
    "https://www.cuotasahora.com/football/africa/copa-de-africa-de-naciones-2025/results/",
    "https://www.cuotasahora.com/football/africa/copa-de-africa-de-naciones-2023/results/",
    "https://www.cuotasahora.com/football/africa/copa-de-africa-de-naciones-2021/results/",
    "https://www.cuotasahora.com/football/africa/copa-de-africa-de-naciones-2019/results/",
    "https://www.cuotasahora.com/football/africa/copa-de-africa-de-naciones-2017/results/",
    "https://www.cuotasahora.com/football/africa/copa-de-africa-de-naciones-2015/results/",

    # Eurocopa
    "https://www.cuotasahora.com/football/europe/eurocopa-2024/results/",
    "https://www.cuotasahora.com/football/europe/eurocopa-2020/results/",
    "https://www.cuotasahora.com/football/europe/eurocopa-2016/results/",
    "https://www.cuotasahora.com/football/europe/eurocopa-2012/results/",
    "https://www.cuotasahora.com/football/europe/eurocopa-2008/results/",

    # Copa América
    "https://www.cuotasahora.com/football/south-america/copa-america/results/",
    "https://www.cuotasahora.com/football/south-america/copa-america-2021/results/",
    "https://www.cuotasahora.com/football/south-america/copa-america-2019/results/",
    "https://www.cuotasahora.com/football/south-america/copa-america-2016/results/",
    "https://www.cuotasahora.com/football/south-america/copa-america-2015/results/",
    "https://www.cuotasahora.com/football/south-america/copa-america-2011/results/",

    # Copa Oro
    "https://www.cuotasahora.com/football/north-central-america/copa-de-oro/results/",
    "https://www.cuotasahora.com/football/north-central-america/copa-de-oro-2023/results/",
    "https://www.cuotasahora.com/football/north-central-america/copa-de-oro-2021/results/",
    "https://www.cuotasahora.com/football/north-central-america/copa-de-oro-2019/results/",
    "https://www.cuotasahora.com/football/north-central-america/copa-de-oro-2017/results/",
    "https://www.cuotasahora.com/football/north-central-america/copa-de-oro-2015/results/",
]

have2pages = [
    # Copa África
    "https://www.cuotasahora.com/football/africa/copa-de-africa-de-naciones-2025/results/",
    "https://www.cuotasahora.com/football/africa/copa-de-africa-de-naciones-2023/results/",
    "https://www.cuotasahora.com/football/africa/copa-de-africa-de-naciones-2021/results/",
    "https://www.cuotasahora.com/football/africa/copa-de-africa-de-naciones-2019/results/",
    "https://www.cuotasahora.com/football/africa/copa-de-africa-de-naciones-2017/results/",
    "https://www.cuotasahora.com/football/africa/copa-de-africa-de-naciones-2015/results/",

    # Eurocopa
    "https://www.cuotasahora.com/football/europe/eurocopa-2024/results/",
    "https://www.cuotasahora.com/football/europe/eurocopa-2020/results/",
    "https://www.cuotasahora.com/football/europe/eurocopa-2016/results/",
    "https://www.cuotasahora.com/football/europe/eurocopa-2012/results/",
    "https://www.cuotasahora.com/football/europe/eurocopa-2008/results/",
]

# Scraper

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install())
)

dataset = []

try:
    existing = pd.read_csv("continental_odds.csv")
    existing["date"] = pd.to_datetime(existing["date"]).dt.strftime("%Y-%m-%d")
    seen = set(zip(existing["date"], existing["home_team"], existing["away_team"]))
    dataset = existing.to_dict("records")
except FileNotFoundError:
    seen = set()
    dataset = []

for i, url in enumerate(urls):
    print(i+1, url)
    pages = [1, 2] if url in have2pages else [1]
    skip_url = False
    for page in pages:
        if skip_url:
            print("Se terminó con la url: ", url)
            break
        paged_url = url if page == 1 else f"{url}#/page/2/"
        try:
            print("Intentando obtener url: ", paged_url)

            driver.get("about:blank") # Leave the page entirely to force navigation
            time.sleep(1)
            driver.get(paged_url)
            time.sleep(6)

            html = BeautifulSoup(driver.page_source, "html.parser")

            current_date = None
            row = 1

            for element in html.select(
                '[data-testid="date-header"], [data-testid="game-row"]'
            ):
                if element.get("data-testid") == "date-header":
                    date_text = element.get_text(strip=True)
                    parts = date_text.split(" - ", 1)
                    if len(parts) == 1:
                        current_date = parse_spanish_date(date_text)
                    else:
                        date_text, stage = parts
                        if stage == "Clasificación" or stage == "Ascenso - Clasificación":
                            skip_url = True
                            break
                        current_date = parse_spanish_date(date_text)

                elif element.get("data-testid") == "game-row":
                    teams = [
                        x.get_text(strip=True)
                        for x in element.select(".participant-name")
                    ]

                    base_odds = [
                        x.get_text(strip=True)
                        for x in element.select(
                            "p[data-testid^='odd-container']"
                        )
                    ]

                    if len(teams) == 2 and len(base_odds) == 3:
                        match_key = (current_date, teams[0], teams[1])
                        if match_key in seen:
                            print(f"{row}: ({current_date}) Skipping already scraped: {teams[0]} vs {teams[1]}")
                            row += 1
                            continue

                        link = element.select_one("a[href]")
                        driver.get("https://www.cuotasahora.com" + link["href"])
                        time.sleep(10)

                        detail_html = BeautifulSoup(
                            driver.page_source,
                            "html.parser"
                        )

                        bookmaker_rows = detail_html.select(
                            '[data-testid="over-under-expanded-row"]'
                        )

                        bookmaker_odds = []

                        for bm_row in bookmaker_rows:
                            try:
                                odds = [
                                    float(x.get_text(strip=True))
                                    for x in bm_row.select("p.odds-text")
                                ]
                            except:
                                continue
                    
                            if len(odds) == 3:
                                bookmaker_odds.append(odds)
                    
                        if not bookmaker_odds:
                            continue

                        bookmaker_odds = np.array(bookmaker_odds)
            
                        dataset.append({
                            "year": int(current_date[:4]),
                            "date": current_date,
                            "home_team": teams[0],
                            "away_team": teams[1],
                            "h_odds_avg": round(bookmaker_odds[:, 0].mean(), 4),
                            "d_odds_avg": round(bookmaker_odds[:, 1].mean(), 4),
                            "a_odds_avg": round(bookmaker_odds[:, 2].mean(), 4),
                        })

                        df = pd.DataFrame(dataset)
                        df["date"] = pd.to_datetime(df["date"])
                        df.to_csv("continental_odds.csv", index=False)

                        seen.add(match_key)

                        print(
                            f"{row}: ({current_date[:4]}) URL {i+1}/{len(urls)}",
                            teams[0],
                            "vs",
                            teams[1]
                        )

                        row += 1
            
        except Exception as e:
            print("FAILED AT URL #", i, ": ", url)
            print(e)
            continue

driver.quit()

# Dataframe

df = pd.DataFrame(dataset)

df["date"] = pd.to_datetime(df["date"])

# Remove potential duplicate matches
df = df.drop_duplicates(
    subset=[
        "date",
        "home_team",
        "away_team"
    ]
).reset_index(drop=True)

df.to_csv("continental_odds.csv", index=False)

print(df.shape)