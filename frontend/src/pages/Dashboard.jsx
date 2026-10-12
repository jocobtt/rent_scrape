import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { trackEvent } from '../lib/posthog';
import InteractiveScatterPlot from '../components/InteractiveScatterPlot';
import Histogram from '../components/Histogram';
import WardHeatmap from '../components/WardHeatmap';
import TokyoMapHeatmap from '../components/TokyoMapHeatmap';
import StatsCards from '../components/StatsCards';
import StatsInsights from '../components/StatsInsights';
import DataFilters from '../components/DataFilters';
import RentByWardChart from '../components/RentByWardChart';
import ApartmentTypeDistribution from '../components/ApartmentTypeDistribution';
import { parseTokyoRentData } from '../lib/dataUtils';
import './Dashboard.css';
import '../styles/cta-button.css';

const DEFAULT_FILTERS = {
  apartmentTypes: [],
  wards: [],
  priceRange: [0, 50],
  sqmRange: [0, 200],
  walkTime: 30,
};

const parseFiltersFromParams = (params) => ({
  apartmentTypes: params.get('types') ? params.get('types').split(',') : [],
  wards: params.get('wards') ? params.get('wards').split(',') : [],
  priceRange: [Number(params.get('priceMin') ?? 0), Number(params.get('priceMax') ?? 50)],
  sqmRange: [Number(params.get('sqmMin') ?? 0), Number(params.get('sqmMax') ?? 200)],
  walkTime: Number(params.get('walk') ?? 30),
});

const filtersToParams = (filters) => {
  const params = {};
  if (filters.apartmentTypes.length) params.types = filters.apartmentTypes.join(',');
  if (filters.wards.length) params.wards = filters.wards.join(',');
  if (filters.priceRange[0] !== 0) params.priceMin = filters.priceRange[0];
  if (filters.priceRange[1] !== 50) params.priceMax = filters.priceRange[1];
  if (filters.sqmRange[0] !== 0) params.sqmMin = filters.sqmRange[0];
  if (filters.sqmRange[1] !== 200) params.sqmMax = filters.sqmRange[1];
  if (filters.walkTime !== 30) params.walk = filters.walkTime;
  return params;
};

const Dashboard = () => {
  const [data, setData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchParams, setSearchParams] = useSearchParams();
  const [filters, setFilters] = useState(() => parseFiltersFromParams(searchParams));

  useEffect(() => {
    const dataUrl = import.meta.env.VITE_DATA_URL || '/tokyo_model.csv';
    fetch(dataUrl)
      .then(response => response.text())
      .then(csvText => {
        const processedData = parseTokyoRentData(csvText);
        setData(processedData);
        setIsLoading(false);
        trackEvent('dashboard_viewed', { total_listings: processedData.length });
      })
      .catch(error => {
        console.error('Error loading CSV:', error);
        setIsLoading(false);
      });
  }, []);

  const filteredData = useMemo(() => {
    let filtered = data;

    if (filters.apartmentTypes.length > 0) {
      filtered = filtered.filter(d => filters.apartmentTypes.includes(d.apartment_type));
    }
    if (filters.wards.length > 0) {
      filtered = filtered.filter(d => filters.wards.includes(d.ku_name));
    }
    filtered = filtered.filter(d =>
      d.rent_price >= filters.priceRange[0] && d.rent_price <= filters.priceRange[1]
    );
    filtered = filtered.filter(d =>
      d.sqr_m >= filters.sqmRange[0] && d.sqr_m <= filters.sqmRange[1]
    );
    filtered = filtered.filter(d => d.eki_walk <= filters.walkTime);

    if (data.length > 0 && filtered.length !== data.length) {
      trackEvent('dashboard_filtered', {
        original_count: data.length,
        filtered_count: filtered.length,
      });
    }

    return filtered;
  }, [filters, data]);

  const handleFilterChange = useCallback((newFilters) => {
    setFilters(prev => {
      const updated = { ...prev, ...newFilters };
      setSearchParams(filtersToParams(updated));
      return updated;
    });
  }, [setSearchParams]);

  const handleResetFilters = useCallback(() => {
    setFilters(DEFAULT_FILTERS);
    setSearchParams({});
    trackEvent('dashboard_filters_reset');
  }, [setSearchParams]);

  if (isLoading) {
    return (
      <div className="dashboard">
        <div className="dashboard-header">
          <h1>🗼 Tokyo Rent Data Dashboard</h1>
        </div>
        <div className="skeleton-stats">
          {[1, 2, 3, 4].map(i => <div key={i} className="skeleton-card shimmer" />)}
        </div>
        <div className="skeleton-chart shimmer" />
      </div>
    );
  }

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>🗼 Tokyo Rent Data Dashboard</h1>
        <p className="dashboard-subtitle">
          Interactive visualization of {data.length.toLocaleString()} rental listings across Tokyo
        </p>
      </div>

      <StatsCards data={filteredData} />
      <StatsInsights data={filteredData} />

      <DataFilters
        data={data}
        filters={filters}
        onFilterChange={handleFilterChange}
        onReset={handleResetFilters}
      />

      {filteredData.length === 0 ? (
        <div className="dashboard-empty">
          <p className="dashboard-empty-icon">🔍</p>
          <h3>No apartments match your filters</h3>
          <p>Try widening your price range, size range, or removing ward/type selections.</p>
          <button className="cta-button" onClick={handleResetFilters}>Reset Filters</button>
        </div>
      ) : (
        <div className="dashboard-grid">
          <div className="chart-container full-width">
            <div className="chart-header">
              <h2>Rent vs Square Meters</h2>
              <p>Hover for details • Click a point to see full listing info</p>
            </div>
            <InteractiveScatterPlot data={filteredData} />
          </div>

          <div className="chart-container full-width">
            <div className="chart-header">
              <h2>Tokyo Geographic Heatmap</h2>
              <p>Average rent prices by ward — hover any ward for details</p>
            </div>
            <TokyoMapHeatmap data={filteredData} width={800} height={700} />
          </div>

          <div className="chart-container full-width">
            <div className="chart-header">
              <h2>Average Rent by Ward — Bar Heatmap</h2>
              <p>Visual comparison of average rent prices across Tokyo&apos;s wards</p>
            </div>
            <WardHeatmap data={filteredData} />
          </div>

          <div className="chart-container">
            <div className="chart-header">
              <h2>Average Rent by Ward</h2>
              <p>Click any bar for min, max, and listing count</p>
            </div>
            <RentByWardChart data={filteredData} />
          </div>

          <div className="chart-container">
            <div className="chart-header">
              <h2>Apartment Type Distribution</h2>
              <p>Breakdown by room configuration</p>
            </div>
            <ApartmentTypeDistribution data={filteredData} />
          </div>

          <div className="chart-container">
            <div className="chart-header">
              <h2>Rent Price Distribution</h2>
              <p>Frequency distribution of monthly rent prices</p>
            </div>
            <Histogram data={filteredData} field="rent_price" numBins={25} width={600} height={400} />
          </div>

          <div className="chart-container">
            <div className="chart-header">
              <h2>Apartment Size Distribution</h2>
              <p>Frequency distribution of apartment sizes</p>
            </div>
            <Histogram data={filteredData} field="sqr_m" numBins={25} width={600} height={400} />
          </div>

          <div className="chart-container">
            <div className="chart-header">
              <h2>Walk Time Distribution</h2>
              <p>Distance to nearest station in minutes</p>
            </div>
            <Histogram data={filteredData} field="eki_walk" numBins={20} width={600} height={400} />
          </div>

          <div className="chart-container">
            <div className="chart-header">
              <h2>Building Age Distribution</h2>
              <p>Years since construction (0 = new build)</p>
            </div>
            <Histogram data={filteredData} field="year_built" numBins={20} width={600} height={400} />
          </div>
        </div>
      )}

      <div className="dashboard-footer">
        <p>
          Showing {filteredData.length.toLocaleString()} of {data.length.toLocaleString()} listings
          {filteredData.length !== data.length && (
            <button className="reset-link" onClick={handleResetFilters}>
              Reset Filters
            </button>
          )}
        </p>
      </div>
    </div>
  );
};

export default Dashboard;
