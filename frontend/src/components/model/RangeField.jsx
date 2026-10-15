import React from 'react';

/** Slider with its current value in the label. Bounds widen to include `value` so a median outside them still shows. */
const RangeField = ({ id, label, unit, value, min, max, step = 1, onChange, format = (v) => v }) => {
  const n = parseFloat(value);
  const lo = Math.min(min, Number.isFinite(n) ? n : min);
  const hi = Math.max(max, Number.isFinite(n) ? n : max);
  return (
    <div className="range-field">
      <label htmlFor={id}>
        {label} <strong>{format(Number.isFinite(n) ? n : lo)}{unit}</strong>
      </label>
      <input
        id={id} type="range" min={lo} max={hi} step={step}
        value={Number.isFinite(n) ? n : lo} onChange={onChange}
      />
    </div>
  );
};

export default RangeField;
