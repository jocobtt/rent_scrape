import { useEffect, useState } from 'react';
import api from '../lib/api';

// One fetch per page load; the summary changes only when a new scrape lands
let cached = null;

/** Ward / layout / station medians from the newest scrape. `data` is null while loading or when unavailable. */
export default function useMarketData() {
  const [state, setState] = useState({ data: cached, loading: !cached });

  useEffect(() => {
    if (cached) return undefined;
    let cancelled = false;
    api.get('/data/market', { timeout: 10000 })
      .then((res) => {
        cached = res.data;
        if (!cancelled) setState({ data: res.data, loading: false });
      })
      .catch(() => { if (!cancelled) setState({ data: null, loading: false }); });
    return () => { cancelled = true; };
  }, []);

  return state;
}
