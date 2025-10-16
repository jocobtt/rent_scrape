import React, { useState, useEffect } from 'react';
import ScatterPlot from '../components/ScatterPlot';
import './ChartView.css';

const ScatterPlotView = () => {
  const [csvData, setCsvData] = useState('');

  useEffect(() => {
    // Fetch CSV data as string
    fetch('/tokyo_model.csv')
      .then(response => response.text())
      .then(data => setCsvData(data))
      .catch(error => console.error('Error loading CSV:', error));
  }, []);

  return (
    <div className="chart-view">
      <h1>Scatter Plot - Age vs Rent Price</h1>
      {csvData && <ScatterPlot csvData={csvData} />}
    </div>
  );
};

export default ScatterPlotView;
