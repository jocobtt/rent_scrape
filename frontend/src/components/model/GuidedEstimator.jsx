import React, { useMemo } from 'react';
import { floorLabel, formatYen } from '../../lib/rentForm';
import { defaultsFor, layoutOptions, compareToTypical, estimateRange } from '../../lib/market';
import useLiveEstimate from '../../hooks/useLiveEstimate';
import { trackEvent } from '../../lib/posthog';
import PlaceSearch from './PlaceSearch';
import ChoiceChips from './ChoiceChips';
import RangeField from './RangeField';
import './GuidedEstimator.css';

const stationOf = (market, values) =>
  market.stations.find((s) => s.name === values.nearestStation && s.line === values.nearestLine) ?? null;

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
 * Pick a place, then a layout; everything else starts at the typical value for that area and
 * the estimate updates as sliders move. The same form values feed the exact-listing form.
 */
const GuidedEstimator = ({ market, values, setField, loadValues, onRefine }) => {
  const station = useMemo(() => stationOf(market, values), [market, values]);
  const layouts = useMemo(() => layoutOptions(market, values.kuName), [market, values.kuName]);
  const hasPlace = Boolean(values.kuName);
  const { estimate, loading, error } = useLiveEstimate(values, hasPlace);

  const popularWards = useMemo(
    () => Object.entries(market.wards)
      .sort((a, b) => b[1].count - a[1].count).slice(0, 8)
      .map(([name, w]) => ({ kind: 'ward', key: `w-${name}`, label: name, ward: name, count: w.count })),
    [market],
  );

  const choosePlace = (place) => {
    // Keep the layout if this ward has it, otherwise start from the ward's most common one
    const opts = layoutOptions(market, place.ward);
    const layout = opts.some((o) => o.value === values.apartmentType) ? values.apartmentType : opts[0]?.value ?? '';
    loadValues({
      kuName: place.ward,
      nearestStation: place.station?.name ?? '',
      nearestLine: place.station?.line ?? '',
      apartmentType: layout,
      ...defaultsFor(market, place.ward, layout, place.station),
    });
    trackEvent('guided_place_selected', { kind: place.kind, ward: place.ward });
  };

  const chooseLayout = (layout) => {
    loadValues({ apartmentType: layout, ...defaultsFor(market, values.kuName, layout, station) });
    trackEvent('guided_layout_selected', { layout });
  };

  const slider = (name) => (e) => setField(name, e.target.value);

  return (
    <div className="guided">
      <PlaceSearch market={market} onSelect={choosePlace} popularWards={hasPlace ? [] : popularWards} />

      {hasPlace && (
        <>
          <div className="guided__block">
            <div id="guided-layout" className="guided__heading">Layout</div>
            <ChoiceChips
              name="layout" labelId="guided-layout" options={layouts}
              value={values.apartmentType} onChange={chooseLayout}
            />
          </div>

          <div className="guided__block guided__sliders">
            <RangeField
              id="g-size" label="Size" unit=" m²" min={15} max={100}
              value={values.sqrM} onChange={slider('sqrM')}
            />
            <RangeField
              id="g-age" label="Building age" unit=" yrs" min={0} max={50}
              value={values.yearBuilt} onChange={slider('yearBuilt')}
              format={(v) => (v === 0 ? 'new' : v)}
            />
            <RangeField
              id="g-walk" label="Walk to station" unit=" min" min={1} max={20}
              value={values.ekiWalk} onChange={slider('ekiWalk')}
            />
            <RangeField
              id="g-floor" label="Floor" min={0} max={30}
              value={values.floor} onChange={slider('floor')} format={(v) => floorLabel(v)}
            />
          </div>

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
