import React from 'react';
import { NumberField, FloorSelect } from './FormFields';

/**
 * The seven fields every prediction needs. Deposit / key money are optional unless
 * `termsRequired` (every method but Standard). `idPrefix` keeps ids unique per page.
 */
const RentFormFields = ({ values, errors, onChange, termsRequired, idPrefix = '' }) => (
  <>
    <NumberField
      id={`${idPrefix}sqrM`} label="Size (sqm):" required
      value={values.sqrM} onChange={onChange('sqrM')} error={errors.sqrM}
      placeholder="e.g. 45" min="5" max="500" step="0.1"
    />
    <NumberField
      id={`${idPrefix}reiPrice`} label="Key Money 礼金 (万円):" required={termsRequired}
      value={values.reiPrice} onChange={onChange('reiPrice')} error={errors.reiPrice}
      placeholder={termsRequired ? 'e.g. 1' : 'e.g. 1 (optional)'} min="0" max="100" step="0.5"
    />
    <NumberField
      id={`${idPrefix}shikikin`} label="Security Deposit 敷金 (万円):" required={termsRequired}
      value={values.shikikin} onChange={onChange('shikikin')} error={errors.shikikin}
      placeholder={termsRequired ? 'e.g. 2' : 'e.g. 2 (optional)'} min="0" max="100" step="0.5"
      hint={!termsRequired && "Optional: leave both blank if you don't know them yet (a street address below improves that estimate)."}
    />
    <NumberField
      id={`${idPrefix}maintenencePrice`} label="Maintenance Fee (万円/month):" required
      value={values.maintenencePrice} onChange={onChange('maintenencePrice')} error={errors.maintenencePrice}
      placeholder="e.g. 0.5" min="0" max="50" step="0.1"
    />
    <NumberField
      id={`${idPrefix}yearBuilt`} label="Building Age (years):" required
      value={values.yearBuilt} onChange={onChange('yearBuilt')} error={errors.yearBuilt}
      placeholder="e.g. 10" min="0" max="100" step="1"
    />
    <FloorSelect
      id={`${idPrefix}floor`} required
      value={values.floor} onChange={onChange('floor')} error={errors.floor}
    />
    <NumberField
      id={`${idPrefix}ekiWalk`} label="Walk to Station (min):" required
      value={values.ekiWalk} onChange={onChange('ekiWalk')} error={errors.ekiWalk}
      placeholder="e.g. 5" min="0" max="120" step="1"
    />
  </>
);

export default RentFormFields;
