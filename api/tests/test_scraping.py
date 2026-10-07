"""
Tests for api/utils/scrape_data.py

Unit tests:   no network required — mocks requests.get with HTML fixtures.
Integration:  marked @pytest.mark.integration — hit real Suumo URL (run manually).

Run unit tests only:
    cd api && pytest tests/test_scraping.py -m "not integration" -v

Run integration smoke test (needs network):
    cd api && pytest tests/test_scraping.py -m integration -v -s
"""

import sys
import os
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

# Ensure api/ is on the path when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.scrape_data import (
    ScrapeData,
    convert_year_built,
    convert_yen_to_number,
    convert_nearest_eki,
    eki_name,
    floor_fix,
)

# ---------------------------------------------------------------------------
# Minimal HTML fixture that mimics one Suumo listing
# ---------------------------------------------------------------------------
MINIMAL_HTML = """
<html><body>
<div class="cassetteitem">
  <div class="cassetteitem-detail">
    <div class="cassetteitem_content">
      <div class="cassetteitem_content-label"><span class="ui-pct ui-pct--util1">賃貸マンション</span></div>
      <div class="cassetteitem_content-title">テストハイツ渋谷</div>
      <ul class="cassetteitem_detail">
        <li class="cassetteitem_detail-col1">東京都渋谷区猿楽町</li>
        <li class="cassetteitem_detail-col2">
          <div class="cassetteitem_detail-text">東急東横線/代官山駅 歩6分</div>
          <div class="cassetteitem_detail-text">ＪＲ山手線/渋谷駅 歩11分</div>
        </li>
        <li class="cassetteitem_detail-col3"><div>築21年</div><div>8階建</div></li>
      </ul>
    </div>
  </div>
  <div class="cassetteitem-item">
    <table class="cassetteitem_other">
      <thead><tr><th></th></tr></thead>
      <tbody>
        <tr class="js-cassette_link">
          <td><input type="checkbox"></td>
          <td><div class="thumb"></div></td>
          <td>4階</td>
          <td><ul>
            <li><span class="cassetteitem_price cassetteitem_price--rent"><span>28万円</span></span></li>
            <li><span class="cassetteitem_price cassetteitem_price--administration">10000円</span></li>
          </ul></td>
          <td><ul>
            <li><span class="cassetteitem_price cassetteitem_price--deposit">28万円</span></li>
            <li><span class="cassetteitem_price cassetteitem_price--gratuity">-</span></li>
          </ul></td>
          <td><ul>
            <li><span class="cassetteitem_madori">1LDK</span></li>
            <li><span class="cassetteitem_menseki">45.13m<sup>2</sup></span></li>
          </ul></td>
        </tr>
        <tr class="js-cassette_link">
          <td><input type="checkbox"></td>
          <td><div class="thumb"></div></td>
          <td>2階</td>
          <td><ul>
            <li><span class="cassetteitem_price cassetteitem_price--rent"><span>12.5万円</span></span></li>
            <li><span class="cassetteitem_price cassetteitem_price--administration">-</span></li>
          </ul></td>
          <td><ul>
            <li><span class="cassetteitem_price cassetteitem_price--deposit">12.5万円</span></li>
            <li><span class="cassetteitem_price cassetteitem_price--gratuity">12.5万円</span></li>
          </ul></td>
          <td><ul>
            <li><span class="cassetteitem_madori">1K</span></li>
            <li><span class="cassetteitem_menseki">22.00m<sup>2</sup></span></li>
          </ul></td>
        </tr>
      </tbody>
    </table>
  </div>
</div>
</body></html>
"""

EMPTY_HTML = "<html><body></body></html>"

SUUMO_URL = (
    "https://suumo.jp/jj/chintai/ichiran/FR301FC001/"
    "?ar=030&bs=040&ta=13&sc=13113&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shkr1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&po2=99"
)


# ---------------------------------------------------------------------------
# Helper: build a mock requests.Response
# ---------------------------------------------------------------------------
def _mock_response(html: str, status_code: int = 200):
    resp = MagicMock()
    resp.text = html
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    return resp


# ===========================================================================
# Unit tests — conversion helpers
# ===========================================================================

class TestConversionHelpers:
    def test_convert_yen_basic(self):
        assert convert_yen_to_number("15万円") == 15.0

    def test_convert_yen_circle(self):
        assert convert_yen_to_number("5000円") == 5000.0

    def test_convert_yen_dash(self):
        assert convert_yen_to_number("-") == 0.0

    def test_convert_year_built_normal(self):
        assert convert_year_built("築10年") == 10

    def test_convert_year_built_new(self):
        assert convert_year_built("新築") == 0

    def test_convert_year_built_dash(self):
        assert convert_year_built("-") == 0

    def test_convert_nearest_eki_with_walk(self):
        assert convert_nearest_eki("渋谷駅 歩5分") == 5

    def test_convert_nearest_eki_no_walk(self):
        assert convert_nearest_eki("-") == 0

    def test_eki_name_with_walk(self):
        assert eki_name("渋谷駅 歩5分") == "渋谷駅 "

    def test_eki_name_no_walk(self):
        assert eki_name("-") == "-"

    def test_floor_fix_basic(self):
        assert floor_fix("3階") == 3

    def test_floor_fix_with_prefix(self):
        assert floor_fix("地上3") == 3


# ===========================================================================
# Unit tests — ScrapeData._safe_text
# ===========================================================================

class TestSafeText:
    def setup_method(self):
        self.scraper = ScrapeData.__new__(ScrapeData)

    def test_safe_text_none_returns_default(self):
        assert self.scraper._safe_text(None) == "-"

    def test_safe_text_none_custom_default(self):
        assert self.scraper._safe_text(None, default="N/A") == "N/A"

    def test_safe_text_real_element(self):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup("<span>  hello  </span>", "html.parser")
        elem = soup.find("span")
        assert self.scraper._safe_text(elem) == "hello"


# ===========================================================================
# Unit tests — ScrapeData.validate_scraped_data
# ===========================================================================

class TestValidateScrapedData:
    def setup_method(self):
        self.scraper = ScrapeData.__new__(ScrapeData)

    def test_passes_with_good_data(self):
        df = pd.DataFrame({
            "rent_price": [10.0] * 20,
            "sqr_m": [45.0] * 20,
        })
        result = self.scraper.validate_scraped_data(df)
        assert result["passed"] is True
        assert result["warnings"] == []

    def test_warns_on_few_rows(self):
        df = pd.DataFrame({"rent_price": [10.0], "sqr_m": [45.0]})
        result = self.scraper.validate_scraped_data(df)
        assert not result["passed"]
        assert any("few listings" in w for w in result["warnings"])

    def test_warns_on_high_zero_rent(self):
        rents = [0.0] * 5 + [10.0] * 10
        df = pd.DataFrame({"rent_price": rents, "sqr_m": [45.0] * 15})
        result = self.scraper.validate_scraped_data(df)
        assert not result["passed"]
        assert any("rent_price" in w for w in result["warnings"])

    def test_warns_on_high_zero_sqrm(self):
        sqrms = [0.0] * 7 + [45.0] * 10
        df = pd.DataFrame({"rent_price": [10.0] * 17, "sqr_m": sqrms})
        result = self.scraper.validate_scraped_data(df)
        assert not result["passed"]
        assert any("sqr_m" in w for w in result["warnings"])

    def test_good_data_is_not_blocking(self):
        df = pd.DataFrame({"rent_price": [10.0] * 20, "sqr_m": [45.0] * 20})
        result = self.scraper.validate_scraped_data(df)
        assert result["blocking"] is False
        assert result["critical"] == []

    def test_blocks_on_too_few_rows(self):
        df = pd.DataFrame({"rent_price": [10.0, 12.0], "sqr_m": [45.0, 50.0]})
        result = self.scraper.validate_scraped_data(df)
        assert result["blocking"] is True
        assert any("Too few listings" in c for c in result["critical"])

    def test_blocks_on_mostly_zero_rent(self):
        rents = [0.0] * 15 + [10.0] * 10
        df = pd.DataFrame({"rent_price": rents, "sqr_m": [45.0] * 25})
        result = self.scraper.validate_scraped_data(df)
        assert result["blocking"] is True
        assert any("rent_price" in c for c in result["critical"])

    def test_blocks_on_mostly_zero_sqrm(self):
        sqrms = [0.0] * 16 + [45.0] * 10
        df = pd.DataFrame({"rent_price": [10.0] * 26, "sqr_m": sqrms})
        result = self.scraper.validate_scraped_data(df)
        assert result["blocking"] is True
        assert any("sqr_m" in c for c in result["critical"])



# ===========================================================================
# Unit tests — ScrapeData.scrape (mocked HTTP)
# ===========================================================================

def _make_scraper_with_mock_session(html: str, pages=(0, 1)):
    """Create a ScrapeData instance whose session.get is mocked."""
    scraper = ScrapeData("https://example.com", pages=pages)
    scraper._session = MagicMock()
    scraper._session.get.return_value = _mock_response(html)
    return scraper


class TestScrapeWithMock:
    def test_scrape_returns_dataframe(self):
        scraper = _make_scraper_with_mock_session(MINIMAL_HTML)
        with patch("utils.scrape_data.sleep"):
            df = scraper.scrape()
        assert isinstance(df, pd.DataFrame)

    def test_scrape_empty_html_returns_empty_df(self):
        scraper = _make_scraper_with_mock_session(EMPTY_HTML)
        with patch("utils.scrape_data.sleep"):
            df = scraper.scrape()
        assert df.empty

    def test_scrape_request_failure_skips_page(self):
        from requests.exceptions import ConnectionError as ReqConnErr
        scraper = ScrapeData("https://example.com", pages=(0, 2))
        scraper._session = MagicMock()
        scraper._session.get.side_effect = ReqConnErr("timeout")
        with patch("utils.scrape_data.sleep"):
            df = scraper.scrape()
        assert df.empty

    def test_retry_adapter_attached(self):
        scraper = ScrapeData("https://example.com", pages=(0, 1))
        adapter = scraper._session.get_adapter("https://")
        assert adapter is not None


# ===========================================================================
# Unit tests — ScrapeData.clean_data (mocked HTTP → then clean)
# ===========================================================================

class TestCleanData:
    def test_clean_data_produces_expected_columns(self):
        scraper = _make_scraper_with_mock_session(MINIMAL_HTML)
        with patch("utils.scrape_data.sleep"):
            raw = scraper.scrape()
        if raw.empty:
            pytest.skip("HTML fixture didn't yield listings — update fixture")
        cleaned = scraper.clean_data(raw)
        for col in ["rent_price", "sqr_m", "year_built", "floor", "nearest_eki_walk"]:
            assert col in cleaned.columns, f"Missing column: {col}"
        # string-only columns should be dropped before training
        for col in ["nearest_eki_name", "direction", "name_place"]:
            assert col not in cleaned.columns, f"Column should be dropped: {col}"

    def test_clean_data_numeric_types(self):
        scraper = _make_scraper_with_mock_session(MINIMAL_HTML)
        with patch("utils.scrape_data.sleep"):
            raw = scraper.scrape()
        if raw.empty:
            pytest.skip("HTML fixture didn't yield listings — update fixture")
        cleaned = scraper.clean_data(raw)
        assert cleaned["year_built"].dtype in [int, "int64", "int32"]
        assert cleaned["floor"].dtype in [int, "int64", "int32"]
        assert cleaned["nearest_eki_walk"].dtype in [int, "int64", "int32"]

    def test_new_fields_present(self):
        scraper = _make_scraper_with_mock_session(MINIMAL_HTML)
        with patch("utils.scrape_data.sleep"):
            raw = scraper.scrape()
        if raw.empty:
            pytest.skip("HTML fixture didn't yield listings — update fixture")
        cleaned = scraper.clean_data(raw)
        assert "contract_type" in cleaned.columns
        assert cleaned["sqr_m"].dtype in [float, "float64", "float32"]


# ===========================================================================
# Integration smoke test — real network (run manually)
# ===========================================================================

@pytest.mark.integration
def test_scrape_one_page_live():
    """
    Verify Suumo is still reachable and parseable.

    Run manually:
        pytest tests/test_scraping.py -m integration -v -s
    """
    scraper = ScrapeData(SUUMO_URL, wait_time_min=1, wait_time_max=3, pages=(0, 1))
    raw = scraper.scrape()
    assert not raw.empty, "No listings parsed — Suumo HTML structure may have changed"

    cleaned = scraper.clean_data(raw)
    assert "rent_price" in cleaned.columns
    assert cleaned["rent_price"].gt(0).any(), "All rent prices are 0 — parse may be broken"
    assert "nearest_eki_walk" in cleaned.columns
    # Core fields must actually be parsed (a silent all-zero column is how a
    # Suumo HTML change broke scraping unnoticed before)
    for col in ("sqr_m", "floor", "year_built"):
        assert cleaned[col].gt(0).mean() > 0.5, f"{col} is mostly zero — selectors may be stale"
    assert cleaned["ku_name"].str.endswith("区").all()
    assert {"building_id", "scraped_at"} <= set(cleaned.columns)

    quality = scraper.validate_scraped_data(cleaned)
    print(f"\nRows parsed: {quality['rows']}")
    print(f"Passed: {quality['passed']}")
    if quality["warnings"]:
        print(f"Warnings: {quality['warnings']}")


# ===========================================================================
# Parsing of the (current) Suumo building block structure
# ===========================================================================

class TestUnitExtraction:
    def _units(self):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(MINIMAL_HTML, "html.parser")
        scraper = ScrapeData(SUUMO_URL, pages=(0, 1))
        return scraper._extract_listings(soup.find("div", class_="cassetteitem"))

    def test_one_row_per_unit(self):
        assert len(self._units()) == 2

    def test_unit_level_fields_differ(self):
        a, b = self._units()
        assert (a["floor"], b["floor"]) == ("4階", "2階")
        assert (a["rent_price"], b["rent_price"]) == ("28万円", "12.5万円")
        assert (a["apartment_type"], b["apartment_type"]) == ("1LDK", "1K")

    def test_building_level_fields_shared(self):
        a, b = self._units()
        assert a["building_id"] == b["building_id"]
        assert a["ku_name"] == b["ku_name"] == "渋谷区"
        assert a["age"] == "築21年"
        assert a["nearest_eki"] == "東急東横線/代官山駅 歩6分"
        assert a["house_type"] == "賃貸マンション"

    def test_area_with_superscript_is_parsed(self):
        from utils.scrape_data import convert_sqr_m
        a, b = self._units()
        assert convert_sqr_m(a["sqr_m"]) == pytest.approx(45.13)
        assert convert_sqr_m(b["sqr_m"]) == pytest.approx(22.0)

    def test_cleaned_unit_rows_have_no_zeroed_core_fields(self):
        scraper = ScrapeData(SUUMO_URL, pages=(0, 1))
        df = pd.DataFrame(self._units())
        clean = scraper.clean_data(df)
        assert (clean["sqr_m"] > 0).all()
        assert (clean["floor"] > 0).all()
        assert (clean["year_built"] == 21).all()
        assert (clean["nearest_eki_walk"] == 6).all()


class TestConvertSqrM:
    @pytest.mark.parametrize("text,expected", [
        ("45.13m2", 45.13), ("45.13m²", 45.13), ("45.13㎡", 45.13), ("45.13m", 45.13), ("30", 30.0),
    ])
    def test_formats(self, text, expected):
        from utils.scrape_data import convert_sqr_m
        assert convert_sqr_m(text) == pytest.approx(expected)

    def test_garbage_is_zero(self):
        from utils.scrape_data import convert_sqr_m
        assert convert_sqr_m("-") == 0.0
