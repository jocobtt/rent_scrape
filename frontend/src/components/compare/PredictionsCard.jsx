import React from 'react';
import { formatYen, formatMan } from '../../lib/rentForm';

const MODEL_META = {
  passed_model:       { label: 'Production (Linear)',   color: '#6366f1' },
  challenger_model:   { label: 'Challenger (LightGBM)', color: '#14b8a6' },
  tokyo_rent_tabpfn:  { label: 'TabPFN',                color: '#f59e0b' },
  tokyo_rent_tabnet:  { label: 'TabNet',                color: '#f97316' },
  tokyo_rent_torch:   { label: 'Custom Neural Net',     color: '#8b5cf6' },
};
const UNKNOWN_MODEL = { color: '#94a3b8' };

// Every model that returned a prediction, lowest first
const buildModelList = (compare) => {
  const rows = [];

  const p = compare.predictions;
  if (p?.passed_model != null) rows.push({ key: 'passed_model', value: p.passed_model });
  if (p?.challenger_model != null) rows.push({ key: 'challenger_model', value: p.challenger_model });

  Object.entries(compare.torch_predictions || {}).forEach(([key, value]) => {
    if (value != null) rows.push({ key, value });
  });

  return rows.sort((a, b) => a.value - b.value);
};

const medianOf = (values) => {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

// "+3.1万 vs median"; blank within 0.05万, where a sign would only be noise
const vsMedian = (value, median) => {
  const diff = value - median;
  if (Math.abs(diff) < 0.05) return 'at the median';
  return `${diff > 0 ? '+' : '−'}${Math.abs(diff).toFixed(1)}万 vs median`;
};

const STAT_CHIPS = (s) => [
  ['Mean', formatYen(s.mean)],
  ['Median', formatYen(s.median)],
  ['Std Dev', formatMan(s.std)],
  ['Range', `${formatYen(s.min)} – ${formatYen(s.max)}`],
  ['Models run', s.total_models],
];

const PredictionsCard = ({ compare, pending = [] }) => {
  const models = buildModelList(compare);
  const stats = compare.comprehensive_statistics;
  const vals = models.map((m) => m.value);
  // Bars start at zero so their lengths are the real proportions; a marker shows the median and
  // the text beside each value says how far from it the model is
  const maxVal = vals.length ? Math.max(...vals) : 1;
  const median = stats?.median ?? medianOf(vals);
  const barWidth = (v) => Math.round((100 * v) / maxVal);
  const medianLeft = barWidth(median);

  return (
    <div className="result-card">
      <h3 className="result-card__title">Predictions</h3>
      <div className="predictions-list">
        {models.map(({ key, value }) => {
          const meta = MODEL_META[key] || { ...UNKNOWN_MODEL, label: key };
          return (
            <div key={key} className="pred-row">
              <span className="pred-row__label">{meta.label}</span>
              <div className="pred-row__track">
                <div
                  className="pred-row__bar"
                  style={{ width: `${barWidth(value)}%`, backgroundColor: meta.color }}
                />
                {models.length > 1 && <span className="pred-row__median" style={{ left: `${medianLeft}%` }} />}
              </div>
              <span className="pred-row__value">
                {formatYen(value)}
                <span className="pred-row__man">
                  {formatMan(value)}{models.length > 1 && ` · ${vsMedian(value, median)}`}
                </span>
              </span>
            </div>
          );
        })}
        {pending.map((key) => (
          <div key={key} className="pred-row pred-row--pending" aria-busy="true">
            <span className="pred-row__label">{(MODEL_META[key] || { label: key }).label}</span>
            <div className="pred-row__track"><div className="pred-row__bar pred-row__bar--pending" /></div>
            <span className="pred-row__value">running…</span>
          </div>
        ))}
      </div>
      {models.length > 1 && <p className="pred-caption">Bars start at zero; the line marks the median of the models.</p>}

      {stats && (
        <div className="stats-strip">
          {STAT_CHIPS(stats).map(([label, value]) => (
            <div key={label} className="stat-chip">
              <span className="stat-chip__label">{label}</span>
              <span className="stat-chip__value">{value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default PredictionsCard;
