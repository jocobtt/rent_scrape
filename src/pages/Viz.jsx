import React, { useState } from 'react';
import ResponsiveLineChart from '../components/ResponsiveLineChart';
import './Viz.css';

const Viz = () => {
  const [data, setData] = useState([10, 40, 15, 25, 50]);

  const addData = () => {
    setData([...data, Math.round(Math.random() * 50)]);
  };

  const filterData = () => {
    setData(data.filter((v) => v <= 35));
  };

  return (
    <div className="viz-page">
      <h1>🗼 Tokyo Apartment Data</h1>
      <ResponsiveLineChart data={data} />
      <div className="buttons">
        <button onClick={addData}>Visualize Data!</button>
        <button onClick={filterData}>Filter data</button>
      </div>
    </div>
  );
};

export default Viz;
