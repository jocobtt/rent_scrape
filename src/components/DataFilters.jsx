import React from 'react';
import { getUniqueValues } from '../lib/dataUtils';
import './DataFilters.css';

const DataFilters = ({ data, filters, onFilterChange, onReset }) => {
  const apartmentTypes = getUniqueValues(data, 'apartment_type');
  const wards = getUniqueValues(data, 'ku_name');

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
        <button className="reset-button" onClick={onReset}>
          Reset All Filters
        </button>
      </div>

      <div className="filter-section">
        <h4>Price Range (万円)</h4>
        <div className="range-inputs">
          <input
            type="number"
            value={filters.priceRange[0]}
            onChange={(e) => onFilterChange({ priceRange: [+e.target.value, filters.priceRange[1]] })}
            min="0"
            max="50"
          />
          <span>to</span>
          <input
            type="number"
            value={filters.priceRange[1]}
            onChange={(e) => onFilterChange({ priceRange: [filters.priceRange[0], +e.target.value] })}
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
            value={filters.sqmRange[0]}
            onChange={(e) => onFilterChange({ sqmRange: [+e.target.value, filters.sqmRange[1]] })}
            min="0"
            max="200"
          />
          <span>to</span>
          <input
            type="number"
            value={filters.sqmRange[1]}
            onChange={(e) => onFilterChange({ sqmRange: [filters.sqmRange[0], +e.target.value] })}
            min="0"
            max="200"
          />
        </div>
      </div>

      <div className="filter-section">
        <h4>Max Walk Time (min): {filters.walkTime}</h4>
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
