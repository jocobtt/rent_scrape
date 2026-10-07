import re
import random
import pandas as pd
import numpy as np
import os
import hashlib
import logging
from bs4 import BeautifulSoup
from time import sleep
from datetime import datetime, timezone
from urllib.parse import urlencode
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
load_dotenv()
# HuggingFace datasets is imported lazily inside upload_to_hf() to keep
# the module lightweight for scraping-only use cases.

logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# ---------------------------------------------------------------------------
# Suumo URL construction helpers
# ---------------------------------------------------------------------------

SUUMO_BASE = "https://suumo.jp/jj/chintai/ichiran/FR301FC001/"

# All 23 Tokyo special wards (特別区) with their Suumo/JIS sc codes.
TOKYO_WARDS: dict[str, str] = {
    "千代田区": "13101",
    "中央区":   "13102",
    "港区":     "13103",
    "新宿区":   "13104",
    "文京区":   "13105",
    "台東区":   "13106",
    "墨田区":   "13107",
    "江東区":   "13108",
    "品川区":   "13109",
    "目黒区":   "13110",
    "大田区":   "13111",
    "世田谷区": "13112",
    "渋谷区":   "13113",
    "中野区":   "13114",
    "杉並区":   "13115",
    "豊島区":   "13116",
    "北区":     "13117",
    "荒川区":   "13118",
    "板橋区":   "13119",
    "練馬区":   "13120",
    "足立区":   "13121",
    "葛飾区":   "13122",
    "江戸川区": "13123",
}

# Convenience: just the codes, in ward order
TOKYO_WARD_CODES: list[str] = list(TOKYO_WARDS.values())


def build_suumo_url(ward_code: str | None = None) -> str:
    """
    Build a Suumo Tokyo rental search URL.

    Args:
        ward_code: Suumo sc code for a specific Tokyo ward (e.g. "13113" for
                   Shibuya-ku).  Pass None to search all of Tokyo.

    Returns:
        Full Suumo search URL string.
    """
    params: dict[str, str] = {
        "ar":    "030",   # Tokyo area
        "bs":    "040",   # Rental
        "ta":    "13",    # Tokyo prefecture
        "cb":    "0.0",   # Min price (unlimited)
        "ct":    "9999999",
        "mb":    "0",     # Min area (unlimited)
        "mt":    "9999999",
        "et":    "9999999",  # Max walk time (unlimited)
        "cn":    "9999999",
        "shkr1": "03",
        "shkr2": "03",
        "shkr3": "03",
        "shkr4": "03",
        "sngz":  "",
        "po1":   "25",
        "po2":   "99",
    }
    if ward_code:
        params["sc"] = ward_code
    return SUUMO_BASE + "?" + urlencode(params)


class ScrapeData:
    def __init__(self, url, wait_time_min=1, wait_time_max=5, pages=(0, 50), output_directory=None):
        self.url = url
        self.wait_time_min = wait_time_min
        self.wait_time_max = wait_time_max
        self.pages = np.arange(*pages)
        self.output_directory = output_directory or os.getcwd()
        self.headers = {
            'accept-language': 'ja,en;q=0.8',
            'User-Agent': random.choice(USER_AGENTS),
        }
        self._failed_count = 0
        # UTC date of this scrape run; stamped on every row so snapshots can be
        # compared over time (time-based splits need at least two snapshots).
        self.scraped_at = datetime.now(timezone.utc).date().isoformat()

        # HTTP session with retry/backoff
        self._session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self._session.mount("https://", HTTPAdapter(max_retries=retries))
        self._session.mount("http://", HTTPAdapter(max_retries=retries))

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def scrape(self):
        """Scrape data from the URL and return a DataFrame."""
        self._failed_count = 0
        rows = []

        for page_num in self.pages:
            try:
                response = self._session.get(
                    f"{self.url}&pn={page_num}",
                    headers=self.headers,
                    timeout=30,
                )
                response.raise_for_status()
            except requests.RequestException as e:
                logger.warning(f"Page {page_num} request failed: {e}")
                continue

            soup = BeautifulSoup(response.text, 'html.parser')
            apartments = soup.find_all('div', class_="cassetteitem")
            sleep(np.random.randint(self.wait_time_min, self.wait_time_max))

            for container in apartments:
                try:
                    for row in self._extract_listings(container):
                        row['scraped_at'] = self.scraped_at
                        rows.append(row)
                except Exception as e:
                    logger.warning(f"Skipping listing on page {page_num}: {e}")
                    self._failed_count += 1
                sleep(np.random.randint(self.wait_time_min, self.wait_time_max))

        logger.info(
            f"Scrape complete: {len(rows)} listings parsed, "
            f"{self._failed_count} failed, {len(self.pages)} pages"
        )
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def clean_data(self, data: pd.DataFrame):
        """Clean scraped data."""
        if data.empty:
            logger.warning("clean_data received an empty DataFrame")
            return data

        # All money fields end up in 万円 (10,000 yen), whatever unit Suumo printed
        data['rent_price'] = data['rent_price'].apply(convert_money_to_man)
        data['sqr_m'] = data['sqr_m'].apply(convert_sqr_m)
        data['rei_price'] = data['rei_price'].apply(convert_money_to_man)
        data['shikikin'] = data['shikikin'].apply(convert_money_to_man)
        data['maintenence_price'] = data['maintenence_price'].apply(convert_money_to_man)
        data['year_built'] = data['age'].apply(convert_year_built)
        data['floor'] = data['floor'].apply(floor_fix)
        data['nearest_eki_walk'] = data['nearest_eki'].apply(convert_nearest_eki)
        data['nearest_eki_name'] = data['nearest_eki'].apply(eki_name)

        # New fields
        data['contract_type'] = data['contract_type'].apply(
            lambda x: 1 if '定期' in str(x) else 0
        )

        data = data.drop(
            columns=['nearest_eki', 'age', 'Lon', 'Lat',
                     'name_place', 'direction', 'nearest_eki_name', 'ward_name'],
            errors='ignore',
        )

        quality = self.validate_scraped_data(data)
        if not quality["passed"]:
            for warning in quality["warnings"]:
                logger.warning(f"[DataQuality] {warning}")

        return data

    def validate_scraped_data(self, df: pd.DataFrame) -> dict:
        """
        Check output data quality. Returns a dict with warnings, critical issues,
        a passed flag (no soft warnings), and a blocking flag (data too broken
        to train on). Called automatically at the end of clean_data().
        """
        warnings = []
        critical = []
        n = len(df)

        if n < 10:
            warnings.append(f"Very few listings parsed: {n}")
        if n < 5:
            critical.append(f"Too few listings to train on: {n} (minimum 5)")

        if n > 0 and "rent_price" in df.columns:
            zero_rent_ratio = df["rent_price"].eq(0).mean()
            if zero_rent_ratio > 0.1:
                warnings.append(">10% of rent_price values are 0 — possible parse failure")
            if zero_rent_ratio > 0.5:
                critical.append(
                    f">50% of rent_price values are 0 ({zero_rent_ratio:.0%}) — scrape likely broken"
                )

        if n > 0 and "sqr_m" in df.columns:
            zero_sqrm_ratio = df["sqr_m"].eq(0).mean()
            if zero_sqrm_ratio > 0.2:
                warnings.append(">20% of sqr_m values are 0")
            if zero_sqrm_ratio > 0.6:
                critical.append(
                    f">60% of sqr_m values are 0 ({zero_sqrm_ratio:.0%}) — scrape likely broken"
                )

        return {
            "rows": n,
            "warnings": warnings,
            "critical": critical,
            "passed": len(warnings) == 0,
            "blocking": len(critical) > 0,
        }

    @classmethod
    def scrape_wards(
        cls,
        ward_codes: list[str] | None = None,
        pages_per_ward: int = 10,
        wait_time_min: int = 2,
        wait_time_max: int = 5,
    ) -> pd.DataFrame:
        """
        Scrape multiple Tokyo wards and return a combined DataFrame.

        Suumo uses 1-indexed pagination (pn=1 is the first page), so this
        method always starts at page 1 regardless of the pages_per_ward value.

        Args:
            ward_codes:     List of Suumo sc codes to scrape.  Defaults to
                            all 23 Tokyo special wards (TOKYO_WARD_CODES).
            pages_per_ward: How many pages to scrape per ward (30 listings/page).
            wait_time_min:  Min seconds to wait between requests.
            wait_time_max:  Max seconds to wait between requests.

        Returns:
            Combined DataFrame with a 'ward_name' column added for provenance.
            Returns an empty DataFrame if nothing could be scraped.
        """
        if ward_codes is None:
            ward_codes = TOKYO_WARD_CODES

        _code_to_name = {v: k for k, v in TOKYO_WARDS.items()}
        all_frames: list[pd.DataFrame] = []

        for code in ward_codes:
            name = _code_to_name.get(code, code)
            url = build_suumo_url(code)
            logger.info("Scraping %s (%s) — %d pages", name, code, pages_per_ward)
            # Suumo pages are 1-indexed; (1, pages_per_ward + 1) → pn=1..N
            scraper = cls(
                url,
                wait_time_min=wait_time_min,
                wait_time_max=wait_time_max,
                pages=(1, pages_per_ward + 1),
            )
            df = scraper.scrape()
            if df.empty:
                logger.warning("  → 0 listings for %s", name)
                continue
            df["ward_name"] = name
            all_frames.append(df)
            logger.info("  → %d listings", len(df))

        if not all_frames:
            logger.error("scrape_wards: no listings collected across all wards")
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)
        logger.info(
            "scrape_wards complete: %d total listings from %d/%d wards",
            len(combined), len(all_frames), len(ward_codes),
        )
        return combined

    def upload_to_hf(self, data: pd.DataFrame, hf_path: str, train_test_split_ratio: float = 0.2):
        """Upload cleaned data to Huggingface."""
        from sklearn.model_selection import train_test_split
        from datasets import load_dataset  # lazy import — only needed here
        train_data, test_data = train_test_split(data, test_size=train_test_split_ratio)
        train_data.to_csv("train.csv", index=False)
        test_data.to_csv("test.csv", index=False)
        data_files = {"train": "train.csv", "test": "test.csv"}
        dataset = load_dataset(hf_path, data_files=data_files)
        return dataset

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _safe_text(self, element, default="-"):
        """Return element.text stripped, or default if element is None."""
        return element.text.strip() if element else default

    def _extract_listings(self, container) -> list[dict]:
        """
        Extract one row per *unit* from a Suumo building block (``div.cassetteitem``).

        A block describes one building and lists several rentable units in a table;
        building-level fields (name, address, stations, age) repeat on every unit
        row and ``building_id`` ties them together. Raises on an unparseable block;
        the caller catches and skips it.
        """
        name_place = self._safe_text(container.find(class_='cassetteitem_content-title'))
        house_type = self._safe_text(container.select_one('.cassetteitem_content-label span'))
        address = self._safe_text(container.find('li', class_='cassetteitem_detail-col1'))
        station_texts = [
            t.get_text(strip=True)
            for t in container.select('li.cassetteitem_detail-col2 .cassetteitem_detail-text')
        ]
        nearest_eki = station_texts[0] if station_texts else "-"
        col3 = container.find('li', class_='cassetteitem_detail-col3')
        col3_divs = [d.get_text(strip=True) for d in col3.find_all('div')] if col3 else []
        age = col3_divs[0] if col3_divs else "-"
        ward = re.search(r'東京都(.+?区)', address)
        ku_name = ward.group(1) if ward else "-"
        building_id = make_building_id(name_place, address)

        rows = []
        for tr in container.select('table.cassetteitem_other tbody tr'):
            tds = tr.find_all('td', recursive=False)
            menseki = tr.find('span', class_='cassetteitem_menseki')
            rows.append({
                'building_id': building_id,
                'rent_price': self._safe_text(tr.find('span', class_='cassetteitem_price--rent')),
                'address': address,
                'sqr_m': ''.join(menseki.find_all(string=True, recursive=False)).strip() if menseki else "-",
                'name_place': name_place,
                'rei_price': self._safe_text(tr.find('span', class_='cassetteitem_price--gratuity')),
                'shikikin': self._safe_text(tr.find('span', class_='cassetteitem_price--deposit')),
                'nearest_eki': nearest_eki,
                'house_type': house_type,
                'floor': tds[2].get_text(strip=True) if len(tds) > 2 else "-",
                'ku_name': ku_name,
                'age': age,
                'apartment_type': self._safe_text(tr.find('span', class_='cassetteitem_madori')),
                'maintenence_price': self._safe_text(tr.find('span', class_='cassetteitem_price--administration')),
                'direction': "-",
                'contract_type': "定期借家" if "定期" in tr.get_text() else "普通借家",
            })
        return rows


# ------------------------------------------------------------------ #
#  Module-level conversion helpers (kept as module functions so they  #
#  can be imported and tested independently)                          #
# ------------------------------------------------------------------ #

def make_building_id(name_place: str, address: str) -> str:
    """Stable id for a building (name + address), used to group listings in splits.

    A building lists several units, and units in one building have near-identical
    rent drivers, so a random split leaks. Grouping by this id keeps all of a
    building's units on one side of a train/test split.
    """
    key = f"{str(name_place).strip()}|{str(address).strip()}".lower()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def convert_year_built(x):
    x = str(x).strip()
    if '新築' in x:
        return 0
    # Prefer explicit 築N年 pattern (handles surrounding context like "地上3階建 築10年")
    match = re.search(r'築(\d+)年', x)
    if match:
        return int(match.group(1))
    # Fallback: strip known characters and parse; return 0 for unparseable values
    # (e.g. "3階建" — building-structure descriptions that land in the age field)
    cleaned = x.replace('年', '').replace('築', '').replace('-', '0').replace('\n', '').strip()
    try:
        return int(cleaned or '0')
    except ValueError:
        return 0


def convert_yen_to_number(x):
    x = str(x).strip()
    cleaned = x.replace('万円', '').replace('円', '').replace('-', '0').strip()
    try:
        return float(cleaned or '0')
    except ValueError:
        return 0.0


def convert_money_to_man(x):
    """Parse a Suumo price into 万円: '28万円' -> 28.0, '10000円' -> 1.0, '-' -> 0.0."""
    x = str(x).strip()
    match = re.search(r'\d+(?:\.\d+)?', x.replace(',', ''))
    if not match:
        return 0.0
    value = float(match.group())
    return value if '万' in x else value / 10000.0


def convert_nearest_eki(x):
    x = str(x)
    if '歩' not in x:
        return 0
    try:
        return int(x.split('歩')[1].split('分')[0])
    except (ValueError, IndexError):
        return 0


def eki_name(x):
    x = str(x)
    return x.split('歩')[0] if '歩' in x else x


def convert_sqr_m(x):
    """Parse area text such as '45.13m2', '45.13m²', '45.13㎡' or '45.13m'."""
    match = re.search(r'\d+(?:\.\d+)?', str(x))
    return float(match.group()) if match else 0.0


def floor_fix(x):
    x = str(x).strip()
    cleaned = x.replace("階", "").replace("地上", "").replace("\n", "").strip()
    try:
        return int(cleaned or '0')
    except ValueError:
        return 0


# Example usage
if __name__ == "__main__":
    url = "https://suumo.jp/jj/chintai/ichiran/FR301FC001/?ar=030&bs=040&..."
    scraper = ScrapeData(url, wait_time_min=1, wait_time_max=3, pages=(0, 50))
    data = scraper.scrape()
    cleaned_data = scraper.clean_data(data)
    public_url = scraper.upload_to_hf(cleaned_data, "jbrazzy/tokyo_rent")
    print("Data uploaded to:", public_url)
