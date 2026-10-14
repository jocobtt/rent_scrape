import React, { useMemo } from 'react';
import { floorLabel } from '../../lib/rentForm';
import { defaultsFor, layoutOptions } from '../../lib/market';
import { trackEvent } from '../../lib/posthog';
import PlaceSearch from './PlaceSearch';
import ChoiceChips from './ChoiceChips';
import RangeField from './RangeField';
import './GuidedEstimator.css';

export const stationOf = (market, values) =>
  market.stations.find((s) => s.name === values.nearestStation && s.line === values.nearestLine) ?? null;

/**
 * Pick a place, then a layout; size, age, walk time and floor start at the typical value for
 * that area. Shared by the Predict Rent and Compare pages; the values live in `useRentForm`.
 */
const GuidedInputs = ({ market, values, setField, loadValues }) => {
  const station = useMemo(() => stationOf(market, values), [market, values]);
  const layouts = useMemo(() => layoutOptions(market, values.kuName), [market, values.kuName]);
  const hasPlace = Boolean(values.kuName);

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
    <>
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
        </>
      )}
    </>
  );
};

export default GuidedInputs;
