import React, { useState, useEffect } from 'react';
import { trackEvent } from '../lib/posthog';
import InteractiveScatterPlot from '../components/InteractiveScatterPlot';
import Histogram from '../components/Histogram';
import WardHeatmap from '../components/WardHeatmap';
import StatsCards from '../components/StatsCards';
import DataFilters from '../components/DataFilters';
import RentByWardChart from '../components/RentByWardChart';
import ApartmentTypeDistribution from '../components/ApartmentTypeDistribution';
import {
  parseTokyoRentData,
  getSummaryStats,
  getUniqueValues,
} from '../lib/dataUtils';
import './Dashboard.css';

const Dashboard = () => {
  const [data, setData] = useState([]);
  const [filteredData, setFilteredData] = useState([]);
  const [filters, setFilters] = useState({
    apartmentTypes: [],
    wards: [],
    priceRange: [0, 50],
    sqmRange: [0, 200],
    walkTime: 30,
  });

  useEffect(() => {
    // Load and process CSV data using dataUtils
    fetch('/tokyo_model.csv')
      .then(response => response.text())
      .then(csvText => {
        const processedData = parseTokyoRentData(csvText);
        setData(processedData);
        setFilteredData(processedData);

        trackEvent('dashboard_viewed', {
          total_listings: processedData.length,
        });
      })
      .catch(error => {
        console.error('Error loading CSV:', error);
      });
  }, []);

  useEffect(() => {
    // Apply filters
    let filtered = [...data];

    if (filters.apartmentTypes.length > 0) {
      filtered = filtered.filter(d =>
        filters.apartmentTypes.includes(d.apartment_type)
      );
    }

    if (filters.wards.length > 0) {
      filtered = filtered.filter(d =>
        filters.wards.includes(d.ku_name)
      );
    }

    filtered = filtered.filter(d =>
      d.rent_price >= filters.priceRange[0] &&
      d.rent_price <= filters.priceRange[1]
    );

    filtered = filtered.filter(d =>
      d.sqr_m >= filters.sqmRange[0] &&
      d.sqr_m <= filters.sqmRange[1]
    );

    filtered = filtered.filter(d =>
      d.eki_walk <= filters.walkTime
    );

    setFilteredData(filtered);

    if (filtered.length !== data.length) {
      trackEvent('dashboard_filtered', {
        original_count: data.length,
        filtered_count: filtered.length,
        filters_applied: Object.keys(filters).filter(key => {
          const value = filters[key];
          return Array.isArray(value) ? value.length > 0 : true;
        }),
      });
    }
  }, [filters, data]);

  const handleFilterChange = (newFilters) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
  };

  const handleResetFilters = () => {
    setFilters({
      apartmentTypes: [],
      wards: [],
      priceRange: [0, 50],
      sqmRange: [0, 200],
      walkTime: 30,
    });
    trackEvent('dashboard_filters_reset');
  };

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>🗼Tokyo Rent Data Dashboard</h1>
        <p className="dashboard-subtitle">
          Interactive visualization of {data.length.toLocaleString()} rental listings across Tokyo
        </p>
      </div>

      <StatsCards data={filteredData} />

      <DataFilters
        data={data}
        filters={filters}
        onFilterChange={handleFilterChange}
        onReset={handleResetFilters}
      />

      <div className="dashboard-grid">
        <div className="chart-container full-width">
          <div className="chart-header">
            <h2>Rent vs Square Meters</h2>
            <p>Click and drag to explore data points • Hover for details</p>
          </div>
          <InteractiveScatterPlot data={filteredData} />
        </div>

        <div className="chart-container full-width">
          <div className="chart-header">
            <h2>Average Rent by Ward - Heatmap</h2>
            <p>Visual comparison of average rent prices across Tokyo's wards</p>
          </div>
          <WardHeatmap data={filteredData} />
        </div>

        <div className="chart-container">
          <div className="chart-header">
            <h2>Average Rent by Ward</h2>
            <p>Tokyo's 23 special wards</p>
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
            <p>Age of buildings in years</p>
          </div>
          <Histogram data={filteredData} field="age" numBins={20} width={600} height={400} />
        </div>
      </div>

      <div className="dashboard-footer">
        <p>
          Showing {filteredData.length} of {data.length} listings
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
