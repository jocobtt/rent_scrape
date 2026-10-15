import React, { useState } from 'react';
import { SelectField, TextField } from '../FormFields';
import { TOKYO_WARDS, APARTMENT_TYPES, HOUSE_TYPES } from '../../lib/rentForm';

const options = (items) => items.map((v) => <option key={v} value={v}>{v}</option>);

/** Collapsible extras that sharpen the estimate: ward, layout, building type, location. */
const OptionalDetails = ({ values, onChange }) => {
  const [open, setOpen] = useState(false);

  return (
    <div className="optional-section">
      <button
        type="button"
        className="optional-toggle"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span>{open ? '▾' : '▸'} Optional details</span>
        <span className="optional-hint">improves accuracy</span>
      </button>

      {open && (
        <div className="optional-fields">
          <SelectField id="kuName" label="Ward 区" value={values.kuName} onChange={onChange('kuName')}>
            <option value="">Any ward</option>
            {options(TOKYO_WARDS)}
          </SelectField>
          <SelectField id="apartmentType" label="Room Layout" value={values.apartmentType} onChange={onChange('apartmentType')}>
            <option value="">Any layout</option>
            {options(APARTMENT_TYPES)}
          </SelectField>
          <SelectField id="houseType" label="Building Type" value={values.houseType} onChange={onChange('houseType')}>
            <option value="">Any type</option>
            {options(HOUSE_TYPES)}
          </SelectField>
          <TextField
            id="address" label="Street address 住所" maxLength={100}
            value={values.address} onChange={onChange('address')}
            placeholder="e.g. 東京都渋谷区猿楽町1"
            hint="Used when deposit / key money are left blank: nearby rents sharpen the estimate."
          />
          <TextField
            id="nearestStation" label="Nearest station 最寄り駅" maxLength={50}
            value={values.nearestStation} onChange={onChange('nearestStation')}
            placeholder="e.g. 代官山 (without 駅)"
          />
          <TextField
            id="nearestLine" label="Line 路線" maxLength={50}
            value={values.nearestLine} onChange={onChange('nearestLine')}
            placeholder="e.g. 東急東横線"
          />
        </div>
      )}
    </div>
  );
};

export default OptionalDetails;
