import React, { useState, useEffect } from 'react';
import api from '../lib/api';
import { trackEvent } from '../lib/posthog';
import './ModelInfo.css';

const ModelInfo = ({ compact = false }) => {
  const [modelData, setModelData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    fetchModelMetadata();
  }, []);

  const fetchModelMetadata = async () => {
    try {
      // Replace with your actual MLflow endpoint
      const response = await api.get('/model/metadata', {
        timeout: 5000,
      });

      setModelData(response.data);
      setIsLoading(false);

      trackEvent('model_metadata_viewed', {
        model_version: response.data.version,
        model_name: response.data.name,
      });
    } catch (err) {
      console.error('Error fetching model metadata:', err);
      setError('Unable to load model information');
      setIsLoading(false);
    }
  };

  const formatDate = (timestamp) => {
    if (!timestamp) return 'N/A';
    return new Date(timestamp).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  const formatMetric = (value) => {
    if (typeof value === 'number') {
      return value.toFixed(4);
    }
    return value;
  };

  const toggleExpanded = () => {
    setIsExpanded(!isExpanded);
    trackEvent('model_info_toggled', {
      expanded: !isExpanded,
    });
  };

  if (isLoading) {
    return (
      <div className={`model-info ${compact ? 'compact' : ''}`}>
        <div className="model-info-loading">Loading model information...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`model-info ${compact ? 'compact' : ''}`}>
        <div className="model-info-error">{error}</div>
      </div>
    );
  }

  if (!modelData) {
    return null;
  }

  // Compact view - just show version and status
  if (compact && !isExpanded) {
    return (
      <div className="model-info compact">
        <button className="model-info-toggle" onClick={toggleExpanded}>
          <span className="model-version-badge">
            Model v{modelData.version || 'N/A'}
          </span>
          <span className="expand-icon">ℹ️</span>
        </button>
      </div>
    );
  }

  // Full view
  return (
    <div className={`model-info ${compact ? 'compact expanded' : ''}`}>
      {compact && (
        <button className="model-info-close" onClick={toggleExpanded}>
          ✕
        </button>
      )}

      <div className="model-info-header">
        <h3>Model Information</h3>
        <span className={`model-status ${modelData.status || 'active'}`}>
          {modelData.status || 'Active'}
        </span>
      </div>

      <div className="model-info-grid">
        {/* Model Identity */}
        <div className="info-section">
          <h4>Model Details</h4>
          <div className="info-row">
            <span className="info-label">Name:</span>
            <span className="info-value">{modelData.name || 'Tokyo Rent Predictor'}</span>
          </div>
          <div className="info-row">
            <span className="info-label">Version:</span>
            <span className="info-value">{modelData.version || 'N/A'}</span>
          </div>
          <div className="info-row">
            <span className="info-label">Run ID:</span>
            <span className="info-value code">{modelData.run_id?.substring(0, 12) || 'N/A'}...</span>
          </div>
          <div className="info-row">
            <span className="info-label">Last Updated:</span>
            <span className="info-value">{formatDate(modelData.last_updated)}</span>
          </div>
        </div>

        {/* Performance Metrics */}
        {modelData.metrics && (
          <div className="info-section">
            <h4>Performance Metrics</h4>
            {Object.entries(modelData.metrics).map(([key, value]) => (
              <div className="info-row" key={key}>
                <span className="info-label">{key.replace(/_/g, ' ').toUpperCase()}:</span>
                <span className="info-value metric">{formatMetric(value)}</span>
              </div>
            ))}
          </div>
        )}

        {/* Model Parameters */}
        {modelData.params && (
          <div className="info-section">
            <h4>Model Parameters</h4>
            {Object.entries(modelData.params).slice(0, 5).map(([key, value]) => (
              <div className="info-row" key={key}>
                <span className="info-label">{key}:</span>
                <span className="info-value">{value}</span>
              </div>
            ))}
          </div>
        )}

        {/* Feature Importance */}
        {modelData.feature_importance && (
          <div className="info-section full-width">
            <h4>Top Features</h4>
            <div className="feature-importance-chart">
              {modelData.feature_importance.slice(0, 5).map((feature, idx) => (
                <div className="feature-bar" key={idx}>
                  <span className="feature-name">{feature.name}</span>
                  <div className="feature-bar-container">
                    <div
                      className="feature-bar-fill"
                      style={{ width: `${(feature.importance * 100).toFixed(1)}%` }}
                    ></div>
                    <span className="feature-importance-value">
                      {(feature.importance * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Additional Info */}
        {modelData.description && (
          <div className="info-section full-width">
            <h4>Description</h4>
            <p className="model-description">{modelData.description}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelInfo;
