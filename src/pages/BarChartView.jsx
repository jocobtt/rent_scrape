import React, { useState, useEffect } from 'react';
import BarChart from '../components/BarChart';
import './ChartView.css';
import * as d3 from 'd3';

const BarChartView = () => {
  const [data, setData] = useState([]);

  useEffect(() => {
    // Fetch CSV data from public directory
    fetch('/tokyo_model.csv')
      .then(response => response.text())
      .then(csvText => {
        const parsedData = d3.csvParse(csvText, d3.autoType);
        setData(parsedData);
      })
      .catch(error => console.error('Error loading CSV:', error));
  }, []);

  return (
    <div className="chart-view">
      <h1>Bar Chart - Tokyo Rent by Ward</h1>
      <BarChart csvData={data} />
    </div>
  );
};

export default BarChartView;
