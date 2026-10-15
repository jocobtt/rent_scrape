// Where the dashboard gets its listings: the API's latest scrape, else the bundled static sample.

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

export const STATIC_SAMPLE_URL = '/tokyo_model.csv';

// VITE_DATA_URL overrides the API (e.g. a CDN copy); the sample is always the last resort
const SOURCES = [
  import.meta.env.VITE_DATA_URL,
  `${API_BASE}/data/listings`,
  STATIC_SAMPLE_URL,
].filter(Boolean);

// A host that rewrites unknown paths to index.html answers 200 with HTML, so check the header row
const looksLikeListings = (text) => text.startsWith('rent_price,');

/**
 * `createdAt` is when the API's scrape was made (null for the bundled sample, whose age is unknown);
 * `isSample` is true when the API had nothing and the bundled copy is shown instead.
 * @returns {Promise<{csvText: string, source: string, createdAt: string|null, isSample: boolean}>}
 */
export async function fetchListingsCsv() {
  for (const url of SOURCES) {
    try {
      const response = await fetch(url);
      if (!response.ok) continue;
      const csvText = await response.text();
      if (looksLikeListings(csvText)) {
        return {
          csvText, source: url, isSample: url === STATIC_SAMPLE_URL,
          createdAt: response.headers.get('X-Dataset-Created'),
        };
      }
    } catch {
      // unreachable: try the next source
    }
  }
  throw new Error('No listings source responded');
}
