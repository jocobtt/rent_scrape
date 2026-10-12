import React, { useState, useEffect } from 'react';
import { getUniqueValues } from '../lib/dataUtils';
import { useToast } from '../context/ToastContext';
import useDebounce from '../hooks/useDebounce';
import './DataFilters.css';

const DataFilters = ({ data, filters, onFilterChange, onReset }) => {
  const { showSuccess } = useToast();
  const apartmentTypes = getUniqueValues(data, 'apartment_type');
  const wards = getUniqueValues(data, 'ku_name');

  const handleCopyLink = () => {
    navigator.clipboard.writeText(window.location.href);
    showSuccess('Filter link copied to clipboard!');
  };

  // Local state for range inputs so typing feels instant; debounce before propagating
  const [localPriceRange, setLocalPriceRange] = useState(filters.priceRange);
  const [localSqmRange, setLocalSqmRange] = useState(filters.sqmRange);

  const debouncedPriceRange = useDebounce(localPriceRange, 300);
  const debouncedSqmRange = useDebounce(localSqmRange, 300);

  useEffect(() => {
    onFilterChange({ priceRange: debouncedPriceRange });
  }, [debouncedPriceRange, onFilterChange]);

  useEffect(() => {
    onFilterChange({ sqmRange: debouncedSqmRange });
  }, [debouncedSqmRange, onFilterChange]);

  // Keep local state in sync when filters are reset externally
  const priceMin = filters.priceRange[0];
  const priceMax = filters.priceRange[1];
  const sqmMin = filters.sqmRange[0];
  const sqmMax = filters.sqmRange[1];

  useEffect(() => {
    setLocalPriceRange([priceMin, priceMax]);
  }, [priceMin, priceMax]);

  useEffect(() => {
    setLocalSqmRange([sqmMin, sqmMax]);
  }, [sqmMin, sqmMax]);

  const handleApartmentTypeChange = (type) => {
    const newTypes = filters.apartmentTypes.includes(type)
      ? filters.apartmentTypes.filter(t => t !== type)
      : [...filters.apartmentTypes, type];
    onFilterChange({ apartmentTypes: newTypes });
  };

  const handleWardChange = (ward) => {
    const newWards = filters.wards.includes(ward)
      ? filters.wards.filter(w => w !== ward)
      : [...filters.wards, ward];
    onFilterChange({ wards: newWards });
  };

  return (
    <div className="data-filters">
      <div className="filter-section">
        <h3>Filters</h3>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button className="reset-button" onClick={onReset}>
            Reset All
          </button>
          <button className="reset-button" onClick={handleCopyLink} title="Copy shareable link with current filters">
            🔗 Share
          </button>
        </div>
      </div>

      <div className="filter-section">
        <h4>Price Range (万円)</h4>
        <div className="range-inputs">
          <input
            type="number"
            value={localPriceRange[0]}
            onChange={(e) => setLocalPriceRange([+e.target.value, localPriceRange[1]])}
            min="0"
            max="50"
          />
          <span>to</span>
          <input
            type="number"
            value={localPriceRange[1]}
            onChange={(e) => setLocalPriceRange([localPriceRange[0], +e.target.value])}
            min="0"
            max="50"
          />
        </div>
      </div>

      <div className="filter-section">
        <h4>Size Range (sqm)</h4>
        <div className="range-inputs">
          <input
            type="number"
            value={localSqmRange[0]}
            onChange={(e) => setLocalSqmRange([+e.target.value, localSqmRange[1]])}
            min="0"
            max="200"
          />
          <span>to</span>
          <input
            type="number"
            value={localSqmRange[1]}
            onChange={(e) => setLocalSqmRange([localSqmRange[0], +e.target.value])}
            min="0"
            max="200"
          />
        </div>
      </div>

      <div className="filter-section">
        <h4>Max Walk Time (min): {filters.walkTime >= 30 ? '30+' : filters.walkTime}</h4>
        <input
          type="range"
          value={filters.walkTime}
          onChange={(e) => onFilterChange({ walkTime: +e.target.value })}
          min="1"
          max="30"
        />
      </div>

      <div className="filter-section">
        <h4>Apartment Type</h4>
        <div className="checkbox-group">
          {apartmentTypes.map(type => (
            <label key={type}>
              <input
                type="checkbox"
                checked={filters.apartmentTypes.includes(type)}
                onChange={() => handleApartmentTypeChange(type)}
              />
              {type}
            </label>
          ))}
        </div>
      </div>

      <div className="filter-section">
        <h4>Ward (Ku)</h4>
        <div className="checkbox-group">
          {wards.map(ward => (
            <label key={ward}>
              <input
                type="checkbox"
                checked={filters.wards.includes(ward)}
                onChange={() => handleWardChange(ward)}
              />
              {ward}
            </label>
          ))}
        </div>
      </div>
    </div>
  );
};

export default DataFilters;
