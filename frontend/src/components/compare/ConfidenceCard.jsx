import React from 'react';
import { formatYen, formatMan } from '../../lib/rentForm';

const CONFIDENCE_CFG = {
  very_high: { label: 'Very High', color: '#10b981' },
  high:      { label: 'High',      color: '#14b8a6' },
  medium:    { label: 'Medium',    color: '#f59e0b' },
  low:       { label: 'Low',       color: '#f97316' },
  very_low:  { label: 'Very Low',  color: '#ef4444' },
};

const AgreementBar = ({ label, count, total, color }) => {
  const pct = total > 0 ? Math.round((count / total) * 100) : 0;
  return (
    <div className="conf-stat">
      <div className="conf-stat__head">
        <span className="conf-stat__label">{label}</span>
        <span className="conf-stat__count">{count} / {total} models</span>
      </div>
      <div className="conf-bar-track">
        <div className="conf-bar-fill" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
};

const MetaItem = ({ label, value }) => (
  <div className="conf-meta-item">
    <span className="conf-meta-label">{label}</span>
    <span className="conf-meta-value">{value}</span>
  </div>
);

const ConfidenceCard = ({ confidence }) => {
  const cfg = CONFIDENCE_CFG[confidence.confidence_level] || CONFIDENCE_CFG.medium;
  const { total_models, within_10_percent, within_5_percent } = confidence.model_agreement;
  const { min, max } = confidence.prediction_range;

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
          <AgreementBar label="Within 10% of mean" count={within_10_percent} total={total_models} color={cfg.color} />
          <AgreementBar label="Within 5% of mean" count={within_5_percent} total={total_models} color={cfg.color} />
        </div>

        <div className="confidence-meta">
          <MetaItem label="Coeff. of variation" value={`${(confidence.coefficient_of_variation * 100).toFixed(1)}%`} />
          <MetaItem label="Std deviation" value={formatMan(confidence.standard_deviation)} />
          <MetaItem label="Prediction range" value={`${formatYen(min)} – ${formatYen(max)}`} />
        </div>
      </div>
    </div>
  );
};

export default ConfidenceCard;
