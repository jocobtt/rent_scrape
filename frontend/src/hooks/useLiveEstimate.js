import { useEffect, useState } from 'react';
import api from '../lib/api';
import useDebounce from './useDebounce';
import { buildRentInput, validateRentForm } from '../lib/rentForm';

/**
 * Re-estimates the rent shortly after the form values settle. Calls are sent with
 * `log=false` so slider-dragging doesn't fill the drift-monitoring log.
 * @returns {{estimate: {prediction:number, variant:string}|null, loading: boolean, error: string|null}}
 */
export default function useLiveEstimate(values, enabled) {
  const settled = useDebounce(values, 500);
  const [state, setState] = useState({ estimate: null, loading: false, error: null });

  useEffect(() => {
    if (!enabled || Object.keys(validateRentForm(settled, { termsRequired: false })).length > 0) {
      setState({ estimate: null, loading: false, error: null });
      return undefined;
    }
    const controller = new AbortController();
    setState((s) => ({ ...s, loading: true, error: null }));
    api.post('/challenger_predict?log=false', buildRentInput(settled), { timeout: 10000, signal: controller.signal })
      .then((res) => setState({ estimate: res.data, loading: false, error: null }))
      .catch((err) => {
        if (controller.signal.aborted) return;
        setState({
          estimate: null, loading: false,
          error: err.response?.data?.detail || 'Could not reach the prediction service.',
        });
      });
    return () => controller.abort();
  }, [settled, enabled]);

  return state;
}
