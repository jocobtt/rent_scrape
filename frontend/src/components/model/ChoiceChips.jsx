import React from 'react';
import '../../styles/chip.css';

/** Single choice as a wrap of pill-shaped radios; `options` are {value, count?}. */
const ChoiceChips = ({ name, labelId, options, value, onChange }) => (
  <div className="chip-group" role="radiogroup" aria-labelledby={labelId}>
    {options.map((o) => (
      <label key={o.value} className="chip">
        <input
          type="radio" name={name} className="chip__input"
          checked={value === o.value} onChange={() => onChange(o.value)}
        />
        <span className="chip__label">
          {o.value}
          {o.count != null && <span className="chip__count">{o.count.toLocaleString()}</span>}
        </span>
      </label>
    ))}
  </div>
);

export default ChoiceChips;
