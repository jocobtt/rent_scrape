import React, { useState } from 'react';

/**
 * Multi-select as a wrap of pill-shaped checkboxes, each with its listing count.
 * Rare options (`isRare`) start collapsed; ones that are selected stay visible.
 */
const ChipGroup = ({ labelId, options, selected, onToggle, isRare = () => false }) => {
  const [showRare, setShowRare] = useState(false);

  const rareCount = options.filter(o => isRare(o)).length;
  const visible = options.filter(o => showRare || !isRare(o) || selected.includes(o.value));

  return (
    <>
      <div className="chip-group" role="group" aria-labelledby={labelId}>
        {visible.map(({ value, count }) => (
          <label key={value} className="chip">
            <input
              type="checkbox"
              className="chip__input"
              checked={selected.includes(value)}
              onChange={() => onToggle(value)}
            />
            <span className="chip__label">
              {value}
              <span className="chip__count">{count.toLocaleString()}</span>
            </span>
          </label>
        ))}
      </div>
      {rareCount > 0 && (
        <button
          type="button"
          className="chip-more"
          onClick={() => setShowRare(v => !v)}
          aria-expanded={showRare}
        >
          {showRare ? 'Hide uncommon' : `Show ${rareCount} uncommon`}
        </button>
      )}
    </>
  );
};

export default ChipGroup;
