import React, { useId, useMemo, useState } from 'react';
import { searchPlaces } from '../../lib/market';

/**
 * Search box for a ward or station, with a suggestion list (ARIA combobox) and a few
 * popular wards to pick without typing. `onSelect` gets a result from `searchPlaces`.
 */
const PlaceSearch = ({ market, onSelect, popularWards = [] }) => {
  const listId = useId();
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);

  const results = useMemo(() => searchPlaces(market, query), [market, query]);
  const showList = open && results.length > 0;

  const choose = (place) => {
    setQuery(place.label);
    setOpen(false);
    onSelect(place);
  };

  const onKeyDown = (e) => {
    if (e.key === 'Escape') {
      setOpen(false);
    } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      setOpen(true);
      const step = e.key === 'ArrowDown' ? 1 : -1;
      setActive((i) => (i + step + results.length) % Math.max(results.length, 1));
    } else if (e.key === 'Enter' && showList && !e.nativeEvent.isComposing) {
      // Enter also confirms an IME conversion; that must not pick a suggestion
      e.preventDefault();
      choose(results[active] ?? results[0]);
    }
  };

  return (
    <div className="place-search">
      <label htmlFor={`${listId}-input`} className="place-search__label">
        Where do you want to live?
      </label>
      <input
        id={`${listId}-input`} type="text" className="form-control"
        role="combobox" aria-expanded={showList} aria-controls={listId} aria-autocomplete="list"
        aria-activedescendant={showList ? `${listId}-${active}` : undefined}
        placeholder="Ward or station, e.g. 渋谷 or 中野"
        autoComplete="off" maxLength={30}
        value={query}
        onChange={(e) => { setQuery(e.target.value); setActive(0); setOpen(true); }}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={onKeyDown}
      />
      {showList && (
        <ul id={listId} role="listbox" className="place-search__list">
          {results.map((p, i) => (
            <li
              key={p.key} id={`${listId}-${i}`} role="option" aria-selected={i === active}
              className={`place-search__option ${i === active ? 'is-active' : ''}`}
              // mousedown, not click: the input's blur would close the list before a click lands
              onMouseDown={(e) => { e.preventDefault(); choose(p); }}
              onMouseEnter={() => setActive(i)}
            >
              <span>{p.label}</span>
              <span className="place-search__meta">
                {p.kind === 'station' ? `${p.station.line} · ${p.ward}` : 'ward'}
              </span>
            </li>
          ))}
        </ul>
      )}
      {popularWards.length > 0 && (
        <div className="place-search__popular" aria-label="Popular wards">
          {popularWards.map((p) => (
            <button key={p.key} type="button" className="place-search__quick" onClick={() => choose(p)}>
              {p.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default PlaceSearch;
