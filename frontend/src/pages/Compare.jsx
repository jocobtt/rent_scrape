import React, { useRef, useState } from 'react';
import api from '../lib/api';
import { useToast } from '../context/ToastContext';
import useRentForm from '../hooks/useRentForm';
import useModelAvailability from '../hooks/useModelAvailability';
import useMarketData from '../hooks/useMarketData';
import { trackEvent } from '../lib/posthog';
import { buildRentInput } from '../lib/rentForm';
import RentFormFields from '../components/RentFormFields';
import { NumberField } from '../components/FormFields';
import GuidedInputs from '../components/model/GuidedInputs';
import ModelStatusBar from '../components/compare/ModelStatusBar';
import PredictionsCard from '../components/compare/PredictionsCard';
import ConfidenceCard from '../components/compare/ConfidenceCard';
import RecommendationCard from '../components/compare/RecommendationCard';
import './Compare.css';

// Deep models, fastest first; each is requested on its own so a row appears when it is ready
const DEEP_MODELS = ['tokyo_rent_tabnet', 'tokyo_rent_tabpfn'];

const settled = (res) => (res.status === 'fulfilled' ? res.value.data : null);

const Compare = () => {
  const { values, errors, onChange, setField, loadValues, setErrors, validate } = useRentForm();
  const market = useMarketData();
  // Without the area medians (no scrape yet, API down) the plain form is used instead
  const guided = market.data !== null;
  const availability = useModelAvailability();
  const { showError } = useToast();

  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null); // { full, compare, confidence, recommendations, partial, pending }
  const runRef = useRef(0); // which submit the in-flight responses belong to

  const handleSubmit = async (e) => {
    e.preventDefault();
    // The ensemble and deep-learning models need both deposit and key money; the linear and
    // LightGBM pair can run without them. The plain form always asks for both.
    const full = !guided || (values.reiPrice !== '' && values.shikikin !== '');
    const found = validate({ termsRequired: !guided });
    if (guided && !full && (values.reiPrice !== '') !== (values.shikikin !== '')) {
      found[values.reiPrice === '' ? 'reiPrice' : 'shikikin'] = 'Enter both deposit and key money, or leave both blank';
      setErrors({ ...found });
    }
    if (Object.keys(found).length > 0) return;

    setIsLoading(true);
    setResults(null);
    runRef.current += 1;

    const input = buildRentInput(values);
    trackEvent('compare_submitted', {
      sqrM: values.sqrM, yearBuilt: values.yearBuilt, floor: values.floor, ekiWalk: values.ekiWalk,
      with_terms: full,
    });

    if (!full) {
      const [res] = await Promise.allSettled([api.post('/compare_predictions', input, { timeout: 15000 })]);
      const compare = settled(res);
      if (!compare) showError('Model comparison failed. Is the API running?');
      finish({ full, compare, confidence: null, recommendations: null, partial: null, pending: [] });
      return;
    }

    // Show each model as soon as it answers: linear + LightGBM and TabNet take well under a
    // second, TabPFN about 10 s. The full set (statistics, confidence, recommendations) replaces
    // the partial rows when it lands; TabPFN's answer is cached server side, so they share its cost.
    const run = ++runRef.current;
    const patch = (fn) => { if (runRef.current === run) setResults((prev) => prev && fn(prev)); };
    setResults({
      full, compare: null, confidence: null, recommendations: null,
      partial: { predictions: null, torch_predictions: {} }, pending: [...DEEP_MODELS],
    });

    const deep = (name) => api.post('/torch/predict?log=false', { model_name: name, input_data: input }, { timeout: 60000 })
      .then((res) => patch((prev) => ({
        ...prev,
        partial: { ...prev.partial, torch_predictions: { ...prev.partial.torch_predictions, [name]: res.data.prediction } },
      })))
      .catch(() => {})
      .finally(() => patch((prev) => ({ ...prev, pending: prev.pending.filter((n) => n !== name) })));

    const [, compareRes, confidenceRes, recsRes] = await Promise.all([
      Promise.all([
        api.post('/compare_predictions', input, { timeout: 15000 })
          .then((res) => patch((prev) => ({ ...prev, partial: { ...prev.partial, predictions: res.data.predictions } })))
          .catch(() => {}),
        ...DEEP_MODELS.map(deep),
      ]),
      ...[
        api.post('/torch/compare-all',        input, { timeout: 60000 }),
        api.post('/ensemble/confidence',      input, { timeout: 30000 }),
        api.post('/ensemble/recommendations', input, { timeout: 30000 }),
      ].map((req) => req.then((value) => ({ status: 'fulfilled', value })).catch(() => ({ status: 'rejected' }))),
    ]);
    if (runRef.current !== run) return;

    const compare = settled(compareRes);
    setResults((prev) => ({
      ...prev, compare, confidence: settled(confidenceRes), recommendations: settled(recsRes), pending: [],
    }));
    setIsLoading(false);
    trackEvent('compare_results', {
      torch_models: compare ? Object.keys(compare.torch_predictions || {}).length : 0,
    });
  };

  // Non-full path: one request, then done
  const finish = (next) => {
    setResults(next);
    setIsLoading(false);
    trackEvent('compare_results', { torch_models: 0 });
  };

  // The complete answer once it lands, the rows that have arrived until then
  const shown = results && (results.compare || (results.partial?.predictions || Object.keys(results.partial?.torch_predictions || {}).length ? results.partial : null));

  return (
    <div className="compare-page">

      <div className="compare-header">
        <h1>Model Comparison</h1>
        <p>Run all available models in parallel and compare their predictions side by side.</p>
      </div>

      {!availability.loading && (
        <ModelStatusBar ensemble={availability.ensemble} torch={availability.torch} />
      )}

      <div className="compare-form-card">
        <form onSubmit={handleSubmit} noValidate>
          {guided ? (
            <>
              <GuidedInputs market={market.data} values={values} setField={setField} loadValues={loadValues} />
              {values.kuName && (
                <fieldset className="compare-listing">
                  <legend>Comparing a specific listing? <span>optional</span></legend>
                  <p className="field-hint">
                    The ensemble and deep-learning models need both deposit and key money. Leave them
                    blank to compare just the linear and LightGBM models.
                  </p>
                  <div className="compare-form-grid compare-listing__grid">
                    <NumberField
                      id="c-maintenencePrice" label="Maintenance (万円/mo):"
                      value={values.maintenencePrice} onChange={onChange('maintenencePrice')} error={errors.maintenencePrice}
                      min="0" max="50" step="0.1"
                    />
                    <NumberField
                      id="c-reiPrice" label="Key Money 礼金 (万円):"
                      value={values.reiPrice} onChange={onChange('reiPrice')} error={errors.reiPrice}
                      placeholder="e.g. 1" min="0" max="100" step="0.5"
                    />
                    <NumberField
                      id="c-shikikin" label="Security Deposit 敷金 (万円):"
                      value={values.shikikin} onChange={onChange('shikikin')} error={errors.shikikin}
                      placeholder="e.g. 2" min="0" max="100" step="0.5"
                    />
                  </div>
                </fieldset>
              )}
            </>
          ) : (
            <div className="compare-form-grid">
              <RentFormFields
                values={values} errors={errors} onChange={onChange}
                termsRequired idPrefix="c-"
              />
            </div>
          )}

          <button
            type="submit" className={`compare-submit ${isLoading ? 'loading' : ''}`}
            disabled={isLoading || (guided && !values.kuName)}
          >
            {isLoading ? 'Running all models...' : 'Compare All Models'}
          </button>
        </form>
      </div>

      {results && (
        <div className="compare-results">
          {!shown && !isLoading && (
            <div className="result-card result-card--error">
              <h3 className="result-card__title">Comparison unavailable</h3>
              <p>The model API didn&apos;t respond. Check the backend is running:</p>
              <code>cd ~/apps/rent_scrape/api &amp;&amp; ./start_api.sh</code>
            </div>
          )}

          {results.compare && !results.full && (
            <div className="result-card result-card--warn">
              <p>
                Compared without deposit and key money, so only the linear and LightGBM models ran.
                Add both above to include the ensemble, TabPFN, TabNet and neural-net models.
              </p>
            </div>
          )}

          {results.compare && results.full && !results.confidence && (
            <div className="result-card result-card--warn">
              <p>Confidence analysis unavailable — ensemble endpoint may be offline.</p>
            </div>
          )}
          {results.compare && results.full && !results.recommendations && (
            <div className="result-card result-card--warn">
              <p>Recommendations unavailable — ensemble endpoint may be offline.</p>
            </div>
          )}

          {shown && <PredictionsCard compare={shown} pending={results.pending} />}
          {results.confidence && <ConfidenceCard confidence={results.confidence} />}
          {results.recommendations && <RecommendationCard recommendations={results.recommendations} />}
        </div>
      )}
    </div>
  );
};

export default Compare;
