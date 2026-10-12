import React from 'react';

const PredictionHistory = ({ history, onClear, onReuse }) => {
  if (history.length === 0) return null;

  return (
    <div className="prediction-history">
      <div className="history-header">
        <h3>Recent Predictions</h3>
        <button className="history-clear-btn" onClick={onClear}>Clear</button>
      </div>
      <div className="history-table-wrapper">
        <table className="history-table">
          <thead>
            <tr>
              <th>Result</th>
              <th>Model</th>
              <th>m²</th>
              <th>Age (yrs)</th>
              <th>Floor</th>
              <th>Walk</th>
              <th>Key 礼金</th>
              <th>Dep. 敷金</th>
              <th>Time</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {history.map((h, i) => (
              <tr key={i}>
                <td><strong>¥{Math.round(h.prediction * 10000).toLocaleString()}</strong></td>
                <td><span className="model-tag small">{h.modelType}</span></td>
                <td>{h.sqrM}</td>
                <td>{h.yearBuilt}</td>
                <td>{h.floor === '0' ? 'G' : `${h.floor}F`}</td>
                <td>{h.ekiWalk}m</td>
                <td>{h.reiPrice === '' ? '—' : `${h.reiPrice}万`}</td>
                <td>{h.shikikin === '' ? '—' : `${h.shikikin}万`}</td>
                <td>{new Date(h.timestamp).toLocaleTimeString()}</td>
                <td>
                  <button className="history-rerun-btn" onClick={() => onReuse(h)}>Re-use</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default PredictionHistory;
