import React from 'react';
import { Link } from 'react-router-dom';
import './Home.css';

const Home = () => {
  return (
    <div className="home-page">
      <div className="hero-section">
        <div className="hero-content">
          <h1 className="title">Predict your rent! 🗼</h1>
          <p className="intro-text">
            This app is for visualizing and predicting rent in the greater Tokyo area!
          </p>
          <Link to="/model">
            <button className="cta-button">Get Started</button>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Home;
