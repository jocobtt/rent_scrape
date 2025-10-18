import React from 'react';
import './About.css';

const About = () => {
  return (
    <div className="about-page">
      <div className="about-container">
        <h1>About This Project</h1>
        <p>
          This is a D3 application for visualizing rent in Tokyo, Japan.
          The data and models are used to help predict apartment rental prices
          in the greater Tokyo area.
        </p>
        <p>
          This application uses React, D3.js for data visualization, and various
          machine learning models to provide accurate rent predictions.
        </p>
      </div>
    </div>
  );
};

export default About;
