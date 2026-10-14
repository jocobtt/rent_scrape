"""Scraped listings for the frontend dashboard."""
from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from config.logging_config import get_logger
from deps import DEFAULT_RATE_LIMIT, limiter
from services import listings_export, market_summary

logger = get_logger(__name__)
router = APIRouter()


@router.get("/data/listings", tags=["Data"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def get_listings(request: Request):
    """
    CSV of the most recent Suumo scrape in the dashboard's columns.

    The dataset hash is the ETag, so a browser re-validates cheaply (304) and picks up a
    new scrape as soon as one is saved. 404 until `scripts/scrape_dataset.py` has run.
    """
    manifest = listings_export.latest_scrape()
    if manifest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No scraped dataset yet. Run scripts/scrape_dataset.py.",
        )
    body, etag = listings_export.dashboard_csv(manifest)
    headers = {"ETag": etag, "Cache-Control": "public, no-cache", "X-Dataset-Created": manifest["created_at"]}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=headers)
    return Response(content=body, media_type="text/csv; charset=utf-8", headers=headers)


@router.get("/data/market", tags=["Data"])
@limiter.limit(DEFAULT_RATE_LIMIT)
async def get_market(request: Request):
    """
    Medians per ward, layout and station from the newest scrape (rents in 万円).

    Starting values and "typical rent" for the prediction form. Same ETag scheme as
    `/data/listings`; 404 until a scrape exists.
    """
    manifest = listings_export.latest_scrape()
    if manifest is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No scraped dataset yet. Run scripts/scrape_dataset.py.",
        )
    etag = f'"market-{manifest["hash_short"]}"'
    headers = {"ETag": etag, "Cache-Control": "public, no-cache"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=headers)
    return JSONResponse(market_summary.market_summary(manifest), headers=headers)
