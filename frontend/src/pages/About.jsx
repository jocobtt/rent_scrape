import React from 'react';
import './About.css';

const About = () => {
  return (
    <div className="about-page">
      <div className="about-container">
        <h1>About This Project</h1>

        <p>
          <strong>Tokyo Rent Explorer</strong> is a full-stack data science project for visualizing
          and predicting apartment rental prices across Tokyo&apos;s 23 special wards. It combines
          real listing data scraped from Suumo with interactive D3.js visualizations and a
          machine learning prediction API.
        </p>

        <h2>The Data</h2>
        <p>
          Rental listings are scraped from <strong>Suumo</strong>, Japan&apos;s largest real estate
          portal, using a custom Python scraper. The dataset covers 900+ apartments across 17 Tokyo
          wards with fields including rent price, apartment type, floor, size (m²), key money,
          security deposit, maintenance fees, and walking distance to the nearest train station.
        </p>

        <h2>Visualizations</h2>
        <p>
          The dashboard is built with <strong>React 18</strong> and <strong>D3.js v7</strong>.
          Charts include an interactive scatter plot (rent vs. size), a geographic heatmap
          overlaid on Tokyo ward boundaries, bar charts, and histograms for rent price,
          apartment size, walk time, and building age. All charts respond to the filter panel
          in real time, and filter state is encoded in the URL for easy sharing.
        </p>

        <h2>Machine Learning</h2>
        <p>
          The rent predictor is served via a <strong>FastAPI</strong> backend with three
          prediction modes. <strong>Standard</strong> uses a production LightGBM gradient
          boosting model. <strong>Ensemble</strong> combines multiple models using weighted
          averaging, simple averaging, stacking, or best-model routing.{' '}
          <strong>Deep Learning</strong> offers TabPFN (a pretrained transformer for tabular
          data), TabNet, and a custom PyTorch neural network.
        </p>
        <p>
          The model pipeline includes automated challenger model evaluation, A/B testing
          between model versions, drift monitoring with Evidently AI, and model registry
          management through <strong>MLflow</strong>. Promotion from staging to production
          requires meeting thresholds on RMSE, R², and MAPE — and at least a 2% improvement
          over the current production model.
        </p>
        <p>
          The <strong>Compare Models</strong> page runs all available models in parallel and
          shows their predictions side by side, along with a confidence analysis (coefficient
          of variation, agreement within 5%/10%) and a recommendation for which model best
          suits the input.
        </p>

        <h2>Tech Stack</h2>
        <p>
          <strong>Frontend:</strong> React 18, D3.js v7, React Router v6, Vite, Axios, PostHog analytics
          <br />
          <strong>Backend:</strong> FastAPI, LightGBM, scikit-learn, MLflow, Evidently AI, Docker
          <br />
          <strong>Data:</strong> Python scraper (BeautifulSoup), pandas, CSV pipeline
        </p>
      </div>
    </div>
  );
};

export default About;
