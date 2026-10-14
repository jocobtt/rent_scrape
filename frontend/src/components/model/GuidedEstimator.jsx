import React, { useMemo } from 'react';
import { formatYen } from '../../lib/rentForm';
import { compareToTypical, estimateRange } from '../../lib/market';
import useLiveEstimate from '../../hooks/useLiveEstimate';
import GuidedInputs, { stationOf } from './GuidedInputs';
import './GuidedEstimator.css';

const EstimateCard = ({ market, values, estimate, loading, error }) => {
  if (error) return <div className="estimate estimate--error" role="alert">{error}</div>;
  if (!estimate) return <div className="estimate estimate--empty" aria-busy={loading}>{loading ? 'Estimating…' : ''}</div>;

  const rent = estimate.prediction;
  const [low, high] = estimateRange(rent, Boolean(values.nearestStation));
  const vs = compareToTypical(market, values.kuName, values.apartmentType, parseFloat(values.sqrM), rent);

  return (
    <div className={`estimate ${loading ? 'is-updating' : ''}`} role="status" aria-live="polite">
      <div className="estimate__label">Estimated rent</div>
      <div className="estimate__value">{formatYen(rent)}<span> / month</span></div>
      <div className="estimate__range">
        typically within {formatYen(low)} – {formatYen(high)}
      </div>
      {vs && (
        <div className="estimate__compare">
          {Math.abs(vs.pct) < 3
            ? 'About the typical rent'
            : `${Math.abs(vs.pct)}% ${vs.pct > 0 ? 'above' : 'below'} the typical rent`}
          {' '}for a {values.apartmentType} in {values.kuName}
          <span className="estimate__sub">Typical {formatYen(vs.typicalRent)} at this size, from {vs.count.toLocaleString()} listings</span>
        </div>
      )}
    </div>
  );
};

/**
 * The Predict Rent page's quick estimate: the shared inputs, then a live estimate that
 * updates as they change. The same form values feed the exact-listing form.
 */
const GuidedEstimator = ({ market, values, setField, loadValues, onRefine }) => {
  const station = useMemo(() => stationOf(market, values), [market, values]);
  const hasPlace = Boolean(values.kuName);
  const { estimate, loading, error } = useLiveEstimate(values, hasPlace);

  return (
    <div className="guided">
      <GuidedInputs market={market} values={values} setField={setField} loadValues={loadValues} />

      {hasPlace && (
        <>
          <EstimateCard market={market} values={values} estimate={estimate} loading={loading} error={error} />

          <p className="guided__note">
            Starting values are typical for {values.kuName} {values.apartmentType}
            {station ? ` near ${station.name}` : ''}. Have a specific listing with deposit and key money?{' '}
            <button type="button" className="link-button" onClick={onRefine}>Enter its exact details</button>
          </p>
        </>
      )}
    </div>
  );
};

export default GuidedEstimator;
