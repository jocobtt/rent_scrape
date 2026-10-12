import React from 'react';
import { Link } from 'react-router-dom';
import './Home.css';

const features = [
  {
    icon: '📊',
    title: 'Explore 900+ Listings',
    description: 'Interactively filter and visualize real Tokyo apartment data across 17 wards with scatter plots, histograms, and bar charts.',
    to: '/dashboard',
  },
  {
    icon: '🗺️',
    title: 'Geographic Heatmap',
    description: 'See average rent prices overlaid on a live map of Tokyo\'s 23 special wards. Hover any ward for detailed stats.',
    to: '/dashboard',
  },
  {
    icon: '🤖',
    title: 'ML Rent Predictor',
    description: 'Get an instant rent estimate powered by a LightGBM model trained on Suumo listings. Enter your apartment details and predict.',
    to: '/model',
  },
];

const Home = () => {
  return (
    <div className="home-page">
      <div className="hero-section">
        <div className="hero-content">
          <h1 className="title">Tokyo Rent Explorer 🗼</h1>
          <p className="intro-text">
            Visualize and predict apartment rental prices across Tokyo using real listing data and machine learning.
          </p>
          <Link to="/dashboard">
            <button className="cta-button">Explore the Data</button>
          </Link>

          <div className="features">
            {features.map((feature) => (
              <Link key={feature.title} to={feature.to} style={{ textDecoration: 'none' }}>
                <div className="feature-card">
                  <div className="feature-icon">{feature.icon}</div>
                  <h3 className="feature-title">{feature.title}</h3>
                  <p className="feature-description">{feature.description}</p>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
