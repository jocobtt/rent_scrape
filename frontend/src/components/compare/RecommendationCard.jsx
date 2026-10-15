import React from 'react';

const modelName = (model) => model.replace(/_/g, ' ');

const RecommendationCard = ({ recommendations }) => {
  const { recommended_models: models, input_analysis: analysis } = recommendations;
  const top = models?.[0];

  return (
    <div className="result-card">
      <h3 className="result-card__title">Recommendation</h3>

      {top && (
        <div className="rec-card">
          <div className="rec-card__top">
            <span className="rec-model-name">{modelName(top.model)}</span>
            <span className={`rec-conf-badge rec-conf-${top.confidence}`}>{top.confidence}</span>
          </div>
          <p className="rec-reason">{top.reason}</p>
        </div>
      )}

      {models?.length > 1 && (
        <div className="rec-others">
          <span className="rec-others__label">Also considered:</span>
          {models.slice(1).map((r, i) => (
            <span key={i} className="rec-other-pill">{modelName(r.model)}</span>
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
};

export default RecommendationCard;
