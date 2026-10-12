import React, { useState, useEffect } from 'react';
import api from '../lib/api';
import { useToast } from '../context/ToastContext';
import { trackEvent } from '../lib/posthog';
import './Compare.css';

const MODEL_META = {
  passed_model:       { label: 'Production (Linear)',    color: '#6366f1' },
  challenger_model:   { label: 'Challenger (LightGBM)',  color: '#14b8a6' },
  tokyo_rent_tabpfn:  { label: 'TabPFN',                 color: '#f59e0b' },
  tokyo_rent_tabnet:  { label: 'TabNet',                 color: '#f97316' },
  tokyo_rent_torch:   { label: 'Custom Neural Net',      color: '#8b5cf6' },
};

const CONFIDENCE_CFG = {
  very_high: { label: 'Very High', color: '#10b981' },
  high:      { label: 'High',      color: '#14b8a6' },
  medium:    { label: 'Medium',    color: '#f59e0b' },
  low:       { label: 'Low',       color: '#f97316' },
  very_low:  { label: 'Very Low',  color: '#ef4444' },
};

const floorOptions = Array.from({ length: 61 }, (_, i) => i);

const fmtYen  = (v) => `¥${Math.round(v * 10000).toLocaleString('ja-JP')}`;
const fmtMan  = (v) => `${v.toFixed(2)} 万円`;

const buildModelList = (compare) => {
  if (!compare) return [];
  const rows = [];

  if (compare.predictions) {
    const p = compare.predictions;
    if (p.passed_model    != null) rows.push({ key: 'passed_model',     value: p.passed_model });
    if (p.challenger_model != null) rows.push({ key: 'challenger_model', value: p.challenger_model });
  }

  if (compare.torch_predictions) {
    Object.entries(compare.torch_predictions).forEach(([key, value]) => {
      if (value != null) rows.push({ key, value });
    });
  }

  return rows.sort((a, b) => a.value - b.value);
};

const Compare = () => {
  // Form
  const [sqrM,              setSqrM]              = useState('');
  const [reiPrice,          setReiPrice]          = useState('');
  const [shikikin,          setShikikin]          = useState('');
  const [maintenencePrice,  setMaintenencePrice]  = useState('');
  const [yearBuilt,         setYearBuilt]         = useState('');
  const [floor,             setFloor]             = useState('');
  const [ekiWalk,           setEkiWalk]           = useState('');
  const [errors,            setErrors]            = useState({});

  // Data
  const [isLoading,       setIsLoading]       = useState(false);
  const [results,         setResults]         = useState(null);
  const [availableModels, setAvailableModels] = useState(null);
  const [modelsLoading,   setModelsLoading]   = useState(true);

  const { showError } = useToast();

  // Load model availability once on mount
  useEffect(() => {
    Promise.allSettled([
      api.get('/torch/models',    { timeout: 5000 }),
      api.get('/ensemble/methods', { timeout: 5000 }),
    ]).then(([torchRes, ensembleRes]) => {
      setAvailableModels({
        torch:    torchRes.status    === 'fulfilled' ? torchRes.value.data    : null,
        ensemble: ensembleRes.status === 'fulfilled' ? ensembleRes.value.data : null,
      });
      setModelsLoading(false);
    });
  }, []);

  const validate = () => {
    const errs = {};
    if (!sqrM            || sqrM            < 5   || sqrM            > 500) errs.sqrM            = 'Enter a valid size (5–500 sqm)';
    if (reiPrice         === '' || reiPrice  < 0   || reiPrice        > 100) errs.reiPrice         = 'Required (0–100 万円)';
    if (shikikin         === '' || shikikin  < 0   || shikikin        > 100) errs.shikikin         = 'Required (0–100 万円)';
    if (maintenencePrice === '' || maintenencePrice < 0 || maintenencePrice > 50) errs.maintenencePrice = 'Required (0–50 万円)';
    if (yearBuilt        === '' || yearBuilt < 0   || yearBuilt       > 100) errs.yearBuilt        = 'Required (0–100 years)';
    if (floor            === '')                                              errs.floor            = 'Select a floor';
    if (!ekiWalk         || ekiWalk          < 0   || ekiWalk         > 120) errs.ekiWalk          = 'Required (0–120 min)';
    setErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const buildInput = () => ({
    sqr_m:             parseFloat(sqrM),
    rei_price:         parseFloat(reiPrice),
    shikikin:          parseFloat(shikikin),
    maintenence_price: parseFloat(maintenencePrice),
    year_built:        parseFloat(yearBuilt),
    floor:             parseFloat(floor),
    eki_walk:          parseFloat(ekiWalk),
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validate()) return;

    setIsLoading(true);
    setResults(null);

    const input = buildInput();
    trackEvent('compare_submitted', { sqrM, yearBuilt, floor, ekiWalk });

    const [compareRes, confidenceRes, recsRes] = await Promise.allSettled([
      api.post('/torch/compare-all',          input, { timeout: 20000 }),
      api.post('/ensemble/confidence',        input, { timeout: 15000 }),
      api.post('/ensemble/recommendations',   input, { timeout: 15000 }),
    ]);

    const next = {
      compare:         compareRes.status  === 'fulfilled' ? compareRes.value.data  : null,
      confidence:      confidenceRes.status === 'fulfilled' ? confidenceRes.value.data : null,
      recommendations: recsRes.status     === 'fulfilled' ? recsRes.value.data     : null,
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

      {/* Model availability status */}
      {!modelsLoading && availableModels && (
        <div className="model-status-bar">
          <span className="status-label">Available:</span>
          <span className="status-pill ok">Base Models (2)</span>
          <span className={`status-pill ${availableModels.ensemble ? 'ok' : 'unavailable'}`}>
            Ensemble ({availableModels.ensemble?.total_methods ?? 0} methods)
          </span>
          <span className={`status-pill ${availableModels.torch?.available ? 'ok' : 'unavailable'}`}>
            Deep Learning ({availableModels.torch?.total_models ?? 0} models)
          </span>
        </div>
      )}

      {/* Input form */}
      <div className="compare-form-card">
        <form onSubmit={handleSubmit} noValidate>
          <div className="compare-form-grid">

            <div className="form-group">
              <label htmlFor="c-sqrM">Size (sqm) <span className="required">*</span></label>
              <input type="number" id="c-sqrM" value={sqrM}
                onChange={e => setSqrM(e.target.value)}
                className={`form-control ${errors.sqrM ? 'error' : ''}`}
                placeholder="e.g. 45" min="5" max="500" step="0.1" />
              {errors.sqrM && <span className="error-message">{errors.sqrM}</span>}
            </div>

            <div className="form-group">
              <label htmlFor="c-reiPrice">Key Money 礼金 (万円) <span className="required">*</span></label>
              <input type="number" id="c-reiPrice" value={reiPrice}
                onChange={e => setReiPrice(e.target.value)}
                className={`form-control ${errors.reiPrice ? 'error' : ''}`}
                placeholder="e.g. 1" min="0" max="100" step="0.5" />
              {errors.reiPrice && <span className="error-message">{errors.reiPrice}</span>}
            </div>

            <div className="form-group">
              <label htmlFor="c-shikikin">Security Deposit 敷金 (万円) <span className="required">*</span></label>
              <input type="number" id="c-shikikin" value={shikikin}
                onChange={e => setShikikin(e.target.value)}
                className={`form-control ${errors.shikikin ? 'error' : ''}`}
                placeholder="e.g. 2" min="0" max="100" step="0.5" />
              {errors.shikikin && <span className="error-message">{errors.shikikin}</span>}
            </div>

            <div className="form-group">
              <label htmlFor="c-maintenance">Maintenance (万円/mo) <span className="required">*</span></label>
              <input type="number" id="c-maintenance" value={maintenencePrice}
                onChange={e => setMaintenencePrice(e.target.value)}
                className={`form-control ${errors.maintenencePrice ? 'error' : ''}`}
                placeholder="e.g. 0.5" min="0" max="50" step="0.1" />
              {errors.maintenencePrice && <span className="error-message">{errors.maintenencePrice}</span>}
            </div>

            <div className="form-group">
              <label htmlFor="c-yearBuilt">Building Age (years) <span className="required">*</span></label>
              <input type="number" id="c-yearBuilt" value={yearBuilt}
                onChange={e => setYearBuilt(e.target.value)}
                className={`form-control ${errors.yearBuilt ? 'error' : ''}`}
                placeholder="e.g. 10" min="0" max="100" step="1" />
              {errors.yearBuilt && <span className="error-message">{errors.yearBuilt}</span>}
            </div>

            <div className="form-group">
              <label htmlFor="c-floor">Floor <span className="required">*</span></label>
              <select id="c-floor" value={floor}
                onChange={e => setFloor(e.target.value)}
                className={`form-control ${errors.floor ? 'error' : ''}`}>
                <option value="">Select floor</option>
                {floorOptions.map(n => (
                  <option key={n} value={n}>{n === 0 ? 'Ground Floor' : `${n}F`}</option>
                ))}
              </select>
              {errors.floor && <span className="error-message">{errors.floor}</span>}
            </div>

            <div className="form-group span-2">
              <label htmlFor="c-ekiWalk">Walk to Station (min) <span className="required">*</span></label>
              <input type="number" id="c-ekiWalk" value={ekiWalk}
                onChange={e => setEkiWalk(e.target.value)}
                className={`form-control ${errors.ekiWalk ? 'error' : ''}`}
                placeholder="e.g. 5" min="0" max="120" step="1" />
              {errors.ekiWalk && <span className="error-message">{errors.ekiWalk}</span>}
            </div>

          </div>

          <button type="submit" className={`compare-submit ${isLoading ? 'loading' : ''}`} disabled={isLoading}>
            {isLoading ? 'Running all models...' : 'Compare All Models'}
          </button>
        </form>
      </div>

      {/* Results */}
      {results && (
        <div className="compare-results">

          {/* API failure error card */}
          {!results.compare && (
            <div className="result-card result-card--error">
              <h3 className="result-card__title">Comparison unavailable</h3>
              <p>The model API didn&apos;t respond. Check the backend is running:</p>
              <code>cd ~/apps/rent_scrape/api &amp;&amp; ./start_api.sh</code>
            </div>
          )}

          {/* Partial-failure notices */}
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

          {/* 1. Prediction bars */}
          {results.compare && (() => {
            const models = buildModelList(results.compare);
            const stats  = results.compare.comprehensive_statistics;
            const vals   = models.map(m => m.value);
            const minVal = Math.min(...vals);
            const maxVal = Math.max(...vals);
            const spread = maxVal - minVal || 1;
            const barW   = v => Math.max(12, Math.round(12 + 88 * (v - minVal) / spread));

            return (
              <div className="result-card">
                <h3 className="result-card__title">Predictions</h3>
                <div className="predictions-list">
                  {models.map(({ key, value }) => {
                    const meta = MODEL_META[key] || { label: key, color: '#94a3b8' };
                    return (
                      <div key={key} className="pred-row">
                        <span className="pred-row__label">{meta.label}</span>
                        <div className="pred-row__track">
                          <div
                            className="pred-row__bar"
                            style={{ width: `${barW(value)}%`, backgroundColor: meta.color }}
                          />
                        </div>
                        <span className="pred-row__value">
                          {fmtYen(value)}
                          <span className="pred-row__man">{fmtMan(value)}</span>
                        </span>
                      </div>
                    );
                  })}
                </div>

                {stats && (
                  <div className="stats-strip">
                    <div className="stat-chip">
                      <span className="stat-chip__label">Mean</span>
                      <span className="stat-chip__value">{fmtYen(stats.mean)}</span>
                    </div>
                    <div className="stat-chip">
                      <span className="stat-chip__label">Median</span>
                      <span className="stat-chip__value">{fmtYen(stats.median)}</span>
                    </div>
                    <div className="stat-chip">
                      <span className="stat-chip__label">Std Dev</span>
                      <span className="stat-chip__value">{fmtMan(stats.std)}</span>
                    </div>
                    <div className="stat-chip">
                      <span className="stat-chip__label">Range</span>
                      <span className="stat-chip__value">{fmtYen(stats.min)} – {fmtYen(stats.max)}</span>
                    </div>
                    <div className="stat-chip">
                      <span className="stat-chip__label">Models run</span>
                      <span className="stat-chip__value">{stats.total_models}</span>
                    </div>
                  </div>
                )}
              </div>
            );
          })()}

          {/* 2. Confidence analysis */}
          {results.confidence && (() => {
            const conf = results.confidence;
            const cfg  = CONFIDENCE_CFG[conf.confidence_level] || CONFIDENCE_CFG.medium;
            const total = conf.model_agreement.total_models;
            const pct10 = total > 0
              ? Math.round(conf.model_agreement.within_10_percent / total * 100)
              : 0;
            const pct5  = total > 0
              ? Math.round(conf.model_agreement.within_5_percent / total * 100)
              : 0;

            return (
              <div className="result-card">
                <h3 className="result-card__title">Model Confidence</h3>
                <div className="confidence-body">
                  <div
                    className="confidence-badge"
                    style={{ color: cfg.color, background: cfg.color + '18', borderColor: cfg.color + '60' }}
                  >
                    {cfg.label} Confidence
                  </div>

                  <div className="confidence-stats">
                    <div className="conf-stat">
                      <div className="conf-stat__head">
                        <span className="conf-stat__label">Within 10% of mean</span>
                        <span className="conf-stat__count">
                          {conf.model_agreement.within_10_percent} / {total} models
                        </span>
                      </div>
                      <div className="conf-bar-track">
                        <div className="conf-bar-fill" style={{ width: `${pct10}%`, backgroundColor: cfg.color }} />
                      </div>
                    </div>

                    <div className="conf-stat">
                      <div className="conf-stat__head">
                        <span className="conf-stat__label">Within 5% of mean</span>
                        <span className="conf-stat__count">
                          {conf.model_agreement.within_5_percent} / {total} models
                        </span>
                      </div>
                      <div className="conf-bar-track">
                        <div className="conf-bar-fill" style={{ width: `${pct5}%`, backgroundColor: cfg.color }} />
                      </div>
                    </div>
                  </div>

                  <div className="confidence-meta">
                    <div className="conf-meta-item">
                      <span className="conf-meta-label">Coeff. of variation</span>
                      <span className="conf-meta-value">{(conf.coefficient_of_variation * 100).toFixed(1)}%</span>
                    </div>
                    <div className="conf-meta-item">
                      <span className="conf-meta-label">Std deviation</span>
                      <span className="conf-meta-value">{fmtMan(conf.standard_deviation)}</span>
                    </div>
                    <div className="conf-meta-item">
                      <span className="conf-meta-label">Prediction range</span>
                      <span className="conf-meta-value">
                        {fmtYen(conf.prediction_range.min)} – {fmtYen(conf.prediction_range.max)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            );
          })()}

          {/* 3. Recommendation */}
          {results.recommendations && (() => {
            const recs     = results.recommendations;
            const topRec   = recs.recommended_models?.[0];
            const analysis = recs.input_analysis;

            return (
              <div className="result-card">
                <h3 className="result-card__title">Recommendation</h3>

                {topRec && (
                  <div className="rec-card">
                    <div className="rec-card__top">
                      <span className="rec-model-name">{topRec.model.replace(/_/g, ' ')}</span>
                      <span className={`rec-conf-badge rec-conf-${topRec.confidence}`}>
                        {topRec.confidence}
                      </span>
                    </div>
                    <p className="rec-reason">{topRec.reason}</p>
                  </div>
                )}

                {recs.recommended_models?.length > 1 && (
                  <div className="rec-others">
                    <span className="rec-others__label">Also considered:</span>
                    {recs.recommended_models.slice(1).map((r, i) => (
                      <span key={i} className="rec-other-pill">{r.model.replace(/_/g, ' ')}</span>
                    ))}
                  </div>
                )}

                {analysis && (
                  <div className="input-analysis">
                    <span className="analysis-pill">Rent tier: <strong>{analysis.rent_category}</strong></span>
                    <span className="analysis-pill">Size: <strong>{analysis.size_category}</strong></span>
                    <span className="analysis-pill">Age: <strong>{analysis.age_category}</strong></span>
                  </div>
                )}
              </div>
            );
          })()}

        </div>
      )}
    </div>
  );
};

export default Compare;
