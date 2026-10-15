import { useEffect, useState } from 'react';
import api from '../lib/api';

/**
 * Which optional model families the API has trained. `ensemble` / `torch` hold the
 * endpoint's response, or null when it failed (nothing trained, API offline).
 */
export default function useModelAvailability() {
  const [state, setState] = useState({ ensemble: null, torch: null, loading: true });

  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([
      api.get('/ensemble/methods', { timeout: 5000 }),
      api.get('/torch/models', { timeout: 5000 }),
    ]).then(([ens, torch]) => {
      if (cancelled) return;
      setState({
        ensemble: ens.status === 'fulfilled' ? ens.value.data : null,
        torch: torch.status === 'fulfilled' ? torch.value.data : null,
        loading: false,
      });
    });
    return () => { cancelled = true; };
  }, []);

  return state;
}
