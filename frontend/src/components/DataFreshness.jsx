import React from 'react';
import './DataFreshness.css';

// Rents move; past this the numbers are worth a caveat
const STALE_AFTER_DAYS = 30;
const DAY_MS = 86400000;

const ageInDays = (createdAt, now) => Math.max(0, Math.floor((now - Date.parse(createdAt)) / DAY_MS));

/**
 * Says where the dashboard's listings come from and how old they are. `isSample` means the API
 * had no scrape and the bundled sample is showing, which can be much older than anything live.
 */
const DataFreshness = ({ createdAt, isSample, now = Date.now() }) => {
  if (isSample) {
    return (
      <p className="data-freshness data-freshness--warn" role="status">
        Showing a bundled sample, not live listings (the data service did not respond), so it may be well out of date.
      </p>
    );
  }
  if (!createdAt || Number.isNaN(Date.parse(createdAt))) return null;

  const days = ageInDays(createdAt, now);
  const date = new Date(createdAt).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
  const stale = days >= STALE_AFTER_DAYS;
  return (
    <p className={`data-freshness ${stale ? 'data-freshness--warn' : ''}`} role="status">
      Listings scraped {date}{days > 0 ? ` (${days} day${days === 1 ? '' : 's'} ago)` : ' (today)'}
      {stale && ' — rents may have moved since'}
    </p>
  );
};

export default DataFreshness;
