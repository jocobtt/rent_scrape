import React, { useState } from 'react';
import api from '../lib/api';
import { useToast } from '../context/ToastContext';
import useRentForm from '../hooks/useRentForm';
import useModelAvailability from '../hooks/useModelAvailability';
import { trackEvent } from '../lib/posthog';
import { buildRentInput } from '../lib/rentForm';
import RentFormFields from '../components/RentFormFields';
import ModelStatusBar from '../components/compare/ModelStatusBar';
import PredictionsCard from '../components/compare/PredictionsCard';
import ConfidenceCard from '../components/compare/ConfidenceCard';
import RecommendationCard from '../components/compare/RecommendationCard';
import './Compare.css';

const settled = (res) => (res.status === 'fulfilled' ? res.value.data : null);

const Compare = () => {
  const { values, errors, onChange, validate } = useRentForm();
  const availability = useModelAvailability();
  const { showError } = useToast();

  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null); // { compare, confidence, recommendations }; null = failed

  const handleSubmit = async (e) => {
    e.preventDefault();
    // Every model is run, so deposit / key money are always required
    if (Object.keys(validate({ termsRequired: true })).length > 0) return;

    setIsLoading(true);
    setResults(null);

    const input = buildRentInput(values);
    trackEvent('compare_submitted', {
      sqrM: values.sqrM, yearBuilt: values.yearBuilt, floor: values.floor, ekiWalk: values.ekiWalk,
    });

    const [compareRes, confidenceRes, recsRes] = await Promise.allSettled([
      api.post('/torch/compare-all',        input, { timeout: 20000 }),
      api.post('/ensemble/confidence',      input, { timeout: 15000 }),
      api.post('/ensemble/recommendations', input, { timeout: 15000 }),
    ]);

    const next = {
      compare: settled(compareRes),
      confidence: settled(confidenceRes),
      recommendations: settled(recsRes),
    };

    if (!next.compare) showError('Model comparison failed. Is the API running?');

    setResults(next);
    setIsLoading(false);

    trackEvent('compare_results', {
      torch_models: next.compare ? Object.keys(next.compare.torch_predictions || {}).length : 0,
      confidence_level: next.confidence?.confidence_level,
    });
  };

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
          <div className="compare-form-grid">
            <RentFormFields
              values={values} errors={errors} onChange={onChange}
              termsRequired idPrefix="c-"
            />
          </div>

          <button type="submit" className={`compare-submit ${isLoading ? 'loading' : ''}`} disabled={isLoading}>
            {isLoading ? 'Running all models...' : 'Compare All Models'}
          </button>
        </form>
      </div>

      {results && (
        <div className="compare-results">
          {!results.compare && (
            <div className="result-card result-card--error">
              <h3 className="result-card__title">Comparison unavailable</h3>
              <p>The model API didn&apos;t respond. Check the backend is running:</p>
              <code>cd ~/apps/rent_scrape/api &amp;&amp; ./start_api.sh</code>
            </div>
          )}

          {results.compare && !results.confidence && (
            <div className="result-card result-card--warn">
              <p>Confidence analysis unavailable — ensemble endpoint may be offline.</p>
            </div>
          )}
          {results.compare && !results.recommendations && (
            <div className="result-card result-card--warn">
              <p>Recommendations unavailable — ensemble endpoint may be offline.</p>
            </div>
          )}

          {results.compare && <PredictionsCard compare={results.compare} />}
          {results.confidence && <ConfidenceCard confidence={results.confidence} />}
          {results.recommendations && <RecommendationCard recommendations={results.recommendations} />}
        </div>
      )}
    </div>
  );
};

export default Compare;
