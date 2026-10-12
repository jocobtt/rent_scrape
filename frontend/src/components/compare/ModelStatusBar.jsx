import React from 'react';

const ModelStatusBar = ({ ensemble, torch }) => (
  <div className="model-status-bar">
    <span className="status-label">Available:</span>
    <span className="status-pill ok">Base Models (2)</span>
    <span className={`status-pill ${ensemble ? 'ok' : 'unavailable'}`}>
      Ensemble ({ensemble?.total_methods ?? 0} methods)
    </span>
    <span className={`status-pill ${torch?.available ? 'ok' : 'unavailable'}`}>
      Deep Learning ({torch?.total_models ?? 0} models)
    </span>
  </div>
);

export default ModelStatusBar;
