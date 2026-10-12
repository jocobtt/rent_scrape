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

const STAT_CHIPS = (s) => [
  ['Mean', formatYen(s.mean)],
  ['Median', formatYen(s.median)],
  ['Std Dev', formatMan(s.std)],
  ['Range', `${formatYen(s.min)} – ${formatYen(s.max)}`],
  ['Models run', s.total_models],
];

const PredictionsCard = ({ compare }) => {
  const models = buildModelList(compare);
  const stats = compare.comprehensive_statistics;
  const vals = models.map((m) => m.value);
  const minVal = Math.min(...vals);
  const spread = Math.max(...vals) - minVal || 1;
  // Bars are scaled to the spread, with a floor so the lowest one stays visible
  const barWidth = (v) => Math.max(12, Math.round(12 + 88 * (v - minVal) / spread));

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
              </div>
              <span className="pred-row__value">
                {formatYen(value)}
                <span className="pred-row__man">{formatMan(value)}</span>
              </span>
            </div>
          );
        })}
      </div>

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
