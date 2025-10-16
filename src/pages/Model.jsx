import React, { useState } from 'react';
import axios from 'axios';
import { useToast } from '../context/ToastContext';
import useRateLimit from '../hooks/useRateLimit';
import { trackEvent } from '../lib/posthog';
import './Model.css';

const Model = () => {
  const [sqrM, setSqrM] = useState('');
  const [apartmentType, setApartmentType] = useState('');
  const [kuName, setKuName] = useState('');
  const [yearBuilt, setYearBuilt] = useState('');
  const [floor, setFloor] = useState('');
  const [rentPrice, setRentPrice] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState({});

  const { showSuccess, showError, showWarning } = useToast();

  const apartmentTypes = ['1DK', '2K', '1K', '2DK', '1LDK', '2LDK', 'Other'];
  const kuNames = ['中野', '目黒', '文京', '新宿', '荒川', '墨田', '中央', '台東', '江東', '品川', '杉並', '港', '大田', 'その他'];
  const yearBuilts = Array.from({ length: 26 }, (_, i) => i.toString());
  const floorOptions = Array.from({ length: 21 }, (_, i) => i);

  // Validate form inputs
  const validateForm = () => {
    const newErrors = {};

    if (!sqrM || sqrM <= 0 || sqrM > 1000) {
      newErrors.sqrM = 'Please enter a valid square meter value (1-1000)';
    }

    if (!apartmentType) {
      newErrors.apartmentType = 'Please select an apartment type';
    }

    if (!kuName) {
      newErrors.kuName = 'Please select a ward (Ku)';
    }

    if (yearBuilt === '') {
      newErrors.yearBuilt = 'Please select the building age';
    }

    if (floor === '') {
      newErrors.floor = 'Please select a floor';
    }

    setErrors(newErrors);
    return { isValid: Object.keys(newErrors).length === 0, errors: newErrors };
  };

  // API call function
  const fetchRentPrediction = async () => {
    try {
      const response = await axios.get('/predict', {
        params: {
          sqrM,
          ApartmentType: apartmentType,
          KuName: kuName,
          Yearbuilt: yearBuilt,
          Floor: floor,
          Distance: '',
          Time: ''
        },
        timeout: 10000 // 10 second timeout
      });
      return response.data;
    } catch (error) {
      if (error.code === 'ECONNABORTED') {
        throw new Error('Request timed out. Please try again.');
      }
      if (error.response) {
        throw new Error(error.response.data?.message || 'Failed to calculate rent. Please try again.');
      }
      throw new Error('Network error. Please check your connection.');
    }
  };

  // Rate-limited API call (5 calls per minute)
  const rateLimitedFetchRentPrediction = useRateLimit(fetchRentPrediction, 5, 60000);

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Track form submission attempt
    trackEvent('rent_calculation_attempted', {
      sqrM,
      apartmentType,
      kuName,
      yearBuilt,
      floor,
    });

    // Validate form
    const { isValid, errors: validationErrors } = validateForm();
    if (!isValid) {
      showWarning('Please fill in all required fields correctly.');
      trackEvent('rent_calculation_validation_failed', {
        errors: Object.keys(validationErrors),
      });
      return;
    }

    setIsLoading(true);
    setRentPrice('');
    setErrors({});

    const startTime = Date.now();

    try {
      const result = await rateLimitedFetchRentPrediction();
      const duration = Date.now() - startTime;

      setRentPrice(result);
      showSuccess('Rent calculated successfully!');

      // Track successful calculation
      trackEvent('rent_calculation_success', {
        sqrM,
        apartmentType,
        kuName,
        yearBuilt,
        floor,
        result,
        duration_ms: duration,
      });
    } catch (error) {
      console.error('Error calculating rent:', error);
      showError(error.message || 'Failed to calculate rent. Please try again.');
      setRentPrice('');

      // Track calculation error
      trackEvent('rent_calculation_error', {
        error: error.message,
        sqrM,
        apartmentType,
        kuName,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="model">
      <div className="model__title">
        <h1>🏡 Calculate your rent!</h1>
        <h2>Input apartment parameters here:</h2>
      </div>
      <div className="model__content">
        <form className="model__form" onSubmit={handleSubmit} noValidate>
          <div className="form-group">
            <label htmlFor="sqrM">
              Square Meters: <span className="required">*</span>
            </label>
            <input
              type="number"
              id="sqrM"
              value={sqrM}
              onChange={(e) => setSqrM(e.target.value)}
              className={`form-control ${errors.sqrM ? 'error' : ''}`}
              placeholder="e.g., 80"
              min="1"
              max="1000"
              required
              aria-required="true"
              aria-invalid={errors.sqrM ? 'true' : 'false'}
              aria-describedby={errors.sqrM ? 'sqrM-error' : undefined}
            />
            {errors.sqrM && (
              <span className="error-message" id="sqrM-error" role="alert">
                {errors.sqrM}
              </span>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="ApartmentType">
              Apartment Type: <span className="required">*</span>
            </label>
            <select
              id="ApartmentType"
              value={apartmentType}
              onChange={(e) => setApartmentType(e.target.value)}
              className={`form-control ${errors.apartmentType ? 'error' : ''}`}
              required
              aria-required="true"
              aria-invalid={errors.apartmentType ? 'true' : 'false'}
            >
              <option value="">Please select one</option>
              {apartmentTypes.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </select>
            {errors.apartmentType && (
              <span className="error-message" role="alert">
                {errors.apartmentType}
              </span>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="KuName">
              Ku Name (Ward): <span className="required">*</span>
            </label>
            <select
              id="KuName"
              value={kuName}
              onChange={(e) => setKuName(e.target.value)}
              className={`form-control ${errors.kuName ? 'error' : ''}`}
              required
              aria-required="true"
              aria-invalid={errors.kuName ? 'true' : 'false'}
            >
              <option value="">Please select one</option>
              {kuNames.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
            {errors.kuName && (
              <span className="error-message" role="alert">
                {errors.kuName}
              </span>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="Yearbuilt">
              Building Age (Years): <span className="required">*</span>
            </label>
            <select
              id="Yearbuilt"
              value={yearBuilt}
              onChange={(e) => setYearBuilt(e.target.value)}
              className={`form-control ${errors.yearBuilt ? 'error' : ''}`}
              required
              aria-required="true"
              aria-invalid={errors.yearBuilt ? 'true' : 'false'}
            >
              <option value="">Please select one</option>
              {yearBuilts.map((year) => (
                <option key={year} value={year}>
                  {year} {year === '0' ? '(New)' : `year${year > 1 ? 's' : ''}`}
                </option>
              ))}
            </select>
            {errors.yearBuilt && (
              <span className="error-message" role="alert">
                {errors.yearBuilt}
              </span>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="Floor">
              Floor: <span className="required">*</span>
            </label>
            <select
              id="Floor"
              value={floor}
              onChange={(e) => setFloor(e.target.value)}
              className={`form-control ${errors.floor ? 'error' : ''}`}
              required
              aria-required="true"
              aria-invalid={errors.floor ? 'true' : 'false'}
            >
              <option value="">Please select one</option>
              {floorOptions.map((floorNum) => (
                <option key={floorNum} value={floorNum}>
                  {floorNum === 0 ? 'Ground Floor' : `${floorNum}${floorNum === 1 ? 'st' : floorNum === 2 ? 'nd' : floorNum === 3 ? 'rd' : 'th'} Floor`}
                </option>
              ))}
            </select>
            {errors.floor && (
              <span className="error-message" role="alert">
                {errors.floor}
              </span>
            )}
          </div>

          <button
            type="submit"
            className={`btn-success ${isLoading ? 'loading' : ''}`}
            disabled={isLoading}
            aria-busy={isLoading}
          >
            {isLoading ? (
              <span>Calculating...</span>
            ) : (
              <span>Calculate Rent!</span>
            )}
          </button>

          {rentPrice && (
            <div className="rentPrice" role="status" aria-live="polite">
              <p>Your estimated rent is: <strong>{rentPrice}</strong></p>
            </div>
          )}

          <p className="disclaimer">
            * This is an estimated rent price based on historical data.
            Actual prices may vary. Limited to 5 calculations per minute.
          </p>
        </form>
      </div>
    </div>
  );
};

export default Model;
