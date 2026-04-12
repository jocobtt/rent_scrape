import React, { useState } from 'react';
import { useToast } from '../context/ToastContext';
import useRateLimit from '../hooks/useRateLimit';
import { trackEvent } from '../lib/posthog';
import api from '../lib/api';
import ModelInfo from '../components/ModelInfo';
import './Model.css';

const floorOptions = Array.from({ length: 61 }, (_, i) => i);

const TOKYO_WARDS = [
  '千代田区', '中央区', '港区', '新宿区', '文京区', '台東区',
  '墨田区', '江東区', '品川区', '目黒区', '大田区', '世田谷区',
  '渋谷区', '中野区', '杉並区', '豊島区', '北区', '荒川区',
  '板橋区', '練馬区', '足立区', '葛飾区', '江戸川区',
];

const APARTMENT_TYPES = [
  'ワンルーム', '1K', '1DK', '1LDK',
  '2K', '2DK', '2LDK',
  '3K', '3DK', '3LDK', '4K以上',
];

const HOUSE_TYPES = [
  '賃貸マンション', '賃貸アパート', '賃貸一戸建て', '賃貸テラスハウス',
];

const ENSEMBLE_METHODS = [
  { value: 'weighted_average', label: 'Weighted Average', desc: 'Optimally weighted combination of all models' },
  { value: 'simple_average',   label: 'Simple Average',   desc: 'Equal-weight average of all models' },
  { value: 'stacking',         label: 'Stacking',         desc: 'Meta-model trained to combine predictions' },
  { value: 'best_model',       label: 'Best Model',       desc: 'Routes to the highest-performing model' },
];

const TORCH_MODELS = [
  { value: 'tokyo_rent_tabpfn', label: 'TabPFN',      desc: 'Pretrained transformer for tabular data' },
  { value: 'tokyo_rent_tabnet', label: 'TabNet',       desc: 'Interpretable attention-based deep learning' },
  { value: 'tokyo_rent_torch',  label: 'Custom NN',    desc: 'Custom PyTorch neural network' },
];

const Model = () => {
  // Core fields
  const [sqrM, setSqrM] = useState('');
  const [reiPrice, setReiPrice] = useState('');
  const [shikikin, setShikikin] = useState('');
  const [maintenencePrice, setMaintenencePrice] = useState('');
  const [yearBuilt, setYearBuilt] = useState('');
  const [floor, setFloor] = useState('');
  const [ekiWalk, setEkiWalk] = useState('');

  // Optional fields
  const [showOptional, setShowOptional] = useState(false);
  const [kuName, setKuName] = useState('');
  const [apartmentType, setApartmentType] = useState('');
  const [houseType, setHouseType] = useState('');
  const [contractType, setContractType] = useState('0');

  // Model selection
  const [predictionMethod, setPredictionMethod] = useState('standard');
  const [ensembleMethod, setEnsembleMethod] = useState('weighted_average');
  const [torchModel, setTorchModel] = useState('tokyo_rent_tabpfn');

  // Results & state
  const [rentPrice, setRentPrice] = useState(null);
  const [resultModelType, setResultModelType] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState({});
  const [history, setHistory] = useState([]);

  const { showSuccess, showError, showWarning } = useToast();

  const validateForm = () => {
    const newErrors = {};
    if (!sqrM || sqrM < 5 || sqrM > 500)
      newErrors.sqrM = 'Enter a valid size (5–500 sqm)';
    if (reiPrice === '' || reiPrice < 0 || reiPrice > 100)
      newErrors.reiPrice = 'Enter key money (0–100 万円)';
    if (shikikin === '' || shikikin < 0 || shikikin > 100)
      newErrors.shikikin = 'Enter security deposit (0–100 万円)';
    if (maintenencePrice === '' || maintenencePrice < 0 || maintenencePrice > 50)
      newErrors.maintenencePrice = 'Enter maintenance fee (0–50 万円)';
    if (yearBuilt === '' || yearBuilt < 0 || yearBuilt > 100)
      newErrors.yearBuilt = 'Enter building age in years (0–100)';
    if (floor === '')
      newErrors.floor = 'Select a floor';
    if (!ekiWalk || ekiWalk < 0 || ekiWalk > 120)
      newErrors.ekiWalk = 'Enter walk time to station (0–120 min)';
    setErrors(newErrors);
    return { isValid: Object.keys(newErrors).length === 0, errors: newErrors };
  };

  const buildBaseData = () => ({
    sqr_m:             parseFloat(sqrM),
    rei_price:         parseFloat(reiPrice),
    shikikin:          parseFloat(shikikin),
    maintenence_price: parseFloat(maintenencePrice),
    year_built:        parseFloat(yearBuilt),
    floor:             parseFloat(floor),
    eki_walk:          parseFloat(ekiWalk),
    ...(kuName        && { ku_name: kuName }),
    ...(apartmentType && { apartment_type: apartmentType }),
    ...(houseType     && { house_type: houseType }),
    contract_type:     parseInt(contractType),
  });

  const fetchRentPrediction = async () => {
    const baseData = buildBaseData();

    if (predictionMethod === 'ensemble') {
      const response = await api.post('/ensemble/predict', {
        input_data: baseData,
        method: ensembleMethod,
      }, { timeout: 15000 });
      return response.data;
    }

    if (predictionMethod === 'torch') {
      const response = await api.post('/torch/predict', {
        input_data: baseData,
        model_name: torchModel,
      }, { timeout: 20000 });
      return response.data;
    }

    const response = await api.post('/predict', baseData, { timeout: 10000 });
    return response.data;
  };

  const rateLimitedFetch = useRateLimit(fetchRentPrediction, 20, 60000);

  const handleSubmit = async (e) => {
    e.preventDefault();

    trackEvent('rent_calculation_attempted', {
      sqrM, yearBuilt, floor, ekiWalk,
      prediction_method: predictionMethod,
      ensemble_method: predictionMethod === 'ensemble' ? ensembleMethod : undefined,
      torch_model: predictionMethod === 'torch' ? torchModel : undefined,
    });

    const { isValid, errors: validationErrors } = validateForm();
    if (!isValid) {
      showWarning('Please fill in all required fields correctly.');
      trackEvent('rent_calculation_validation_failed', { errors: Object.keys(validationErrors) });
      return;
    }

    setIsLoading(true);
    setRentPrice(null);
    setResultModelType(null);
    setErrors({});

    const startTime = Date.now();

    try {
      const result = await rateLimitedFetch();
      const duration = Date.now() - startTime;

      setRentPrice(result.prediction);
      setResultModelType(result.model_type || result.model_name || predictionMethod);
      setHistory(prev => [{
        prediction: result.prediction,
        modelType: result.model_type || result.model_name || predictionMethod,
        sqrM, yearBuilt, floor, ekiWalk, reiPrice, shikikin,
        timestamp: new Date().toLocaleTimeString(),
      }, ...prev].slice(0, 5));
      showSuccess('Rent calculated successfully!');

      trackEvent('rent_calculation_success', {
        sqrM, yearBuilt, floor, ekiWalk,
        prediction: result.prediction,
        model_type: result.model_type,
        prediction_method: predictionMethod,
        duration_ms: duration,
      });
    } catch (error) {
      console.error('Error calculating rent:', error);
      const msg = error.response?.data?.detail || error.message || 'Failed to calculate rent. Please try again.';
      showError(msg);
      trackEvent('rent_calculation_error', { error: error.message, sqrM, yearBuilt, prediction_method: predictionMethod });
    } finally {
      setIsLoading(false);
    }
  };

  const formatPrediction = (value) => {
    const yen = Math.round(value * 10000);
    return `¥${yen.toLocaleString('ja-JP')} / month (${value.toFixed(2)} 万円)`;
  };

  return (
    <div className="model">
      <div className="model__title">
        <h1>Calculate your rent!</h1>
        <h2>Input apartment parameters:</h2>
      </div>

      <div className="model__content">
        <form className="model__form" onSubmit={handleSubmit} noValidate>

          {/* ── Required fields ── */}
          <div className="form-group">
            <label htmlFor="sqrM">Size (sqm): <span className="required">*</span></label>
            <input
              type="number" id="sqrM" value={sqrM}
              onChange={(e) => setSqrM(e.target.value)}
              className={`form-control ${errors.sqrM ? 'error' : ''}`}
              placeholder="e.g. 45" min="5" max="500" step="0.1" required
              aria-required="true" aria-invalid={errors.sqrM ? 'true' : 'false'}
            />
            {errors.sqrM && <span className="error-message" role="alert">{errors.sqrM}</span>}
          </div>

          <div className="form-group">
            <label htmlFor="reiPrice">Key Money 礼金 (万円): <span className="required">*</span></label>
            <input
              type="number" id="reiPrice" value={reiPrice}
              onChange={(e) => setReiPrice(e.target.value)}
              className={`form-control ${errors.reiPrice ? 'error' : ''}`}
              placeholder="e.g. 1" min="0" max="100" step="0.5" required
              aria-required="true" aria-invalid={errors.reiPrice ? 'true' : 'false'}
            />
            {errors.reiPrice && <span className="error-message" role="alert">{errors.reiPrice}</span>}
          </div>

          <div className="form-group">
            <label htmlFor="shikikin">Security Deposit 敷金 (万円): <span className="required">*</span></label>
            <input
              type="number" id="shikikin" value={shikikin}
              onChange={(e) => setShikikin(e.target.value)}
              className={`form-control ${errors.shikikin ? 'error' : ''}`}
              placeholder="e.g. 2" min="0" max="100" step="0.5" required
              aria-required="true" aria-invalid={errors.shikikin ? 'true' : 'false'}
            />
            {errors.shikikin && <span className="error-message" role="alert">{errors.shikikin}</span>}
          </div>

          <div className="form-group">
            <label htmlFor="maintenencePrice">Maintenance Fee (万円/month): <span className="required">*</span></label>
            <input
              type="number" id="maintenencePrice" value={maintenencePrice}
              onChange={(e) => setMaintenencePrice(e.target.value)}
              className={`form-control ${errors.maintenencePrice ? 'error' : ''}`}
              placeholder="e.g. 0.5" min="0" max="50" step="0.1" required
              aria-required="true" aria-invalid={errors.maintenencePrice ? 'true' : 'false'}
            />
            {errors.maintenencePrice && <span className="error-message" role="alert">{errors.maintenencePrice}</span>}
          </div>

          <div className="form-group">
            <label htmlFor="yearBuilt">Building Age (years): <span className="required">*</span></label>
            <input
              type="number" id="yearBuilt" value={yearBuilt}
              onChange={(e) => setYearBuilt(e.target.value)}
              className={`form-control ${errors.yearBuilt ? 'error' : ''}`}
              placeholder="e.g. 10" min="0" max="100" step="1" required
              aria-required="true" aria-invalid={errors.yearBuilt ? 'true' : 'false'}
            />
            {errors.yearBuilt && <span className="error-message" role="alert">{errors.yearBuilt}</span>}
          </div>

          <div className="form-group">
            <label htmlFor="floor">Floor: <span className="required">*</span></label>
            <select
              id="floor" value={floor}
              onChange={(e) => setFloor(e.target.value)}
              className={`form-control ${errors.floor ? 'error' : ''}`}
              required aria-required="true" aria-invalid={errors.floor ? 'true' : 'false'}
            >
              <option value="">Select floor</option>
              {floorOptions.map((n) => (
                <option key={n} value={n}>{n === 0 ? 'Ground Floor' : `${n}F`}</option>
              ))}
            </select>
            {errors.floor && <span className="error-message" role="alert">{errors.floor}</span>}
          </div>

          <div className="form-group">
            <label htmlFor="ekiWalk">Walk to Station (min): <span className="required">*</span></label>
            <input
              type="number" id="ekiWalk" value={ekiWalk}
              onChange={(e) => setEkiWalk(e.target.value)}
              className={`form-control ${errors.ekiWalk ? 'error' : ''}`}
              placeholder="e.g. 5" min="0" max="120" step="1" required
              aria-required="true" aria-invalid={errors.ekiWalk ? 'true' : 'false'}
            />
            {errors.ekiWalk && <span className="error-message" role="alert">{errors.ekiWalk}</span>}
          </div>

          {/* ── Optional fields ── */}
          <div className="optional-section">
            <button
              type="button"
              className="optional-toggle"
              onClick={() => setShowOptional(v => !v)}
              aria-expanded={showOptional}
            >
              <span>{showOptional ? '▾' : '▸'} Optional details</span>
              <span className="optional-hint">improves accuracy</span>
            </button>

            {showOptional && (
              <div className="optional-fields">
                <div className="form-group">
                  <label htmlFor="kuName">Ward 区</label>
                  <select
                    id="kuName" value={kuName}
                    onChange={(e) => setKuName(e.target.value)}
                    className="form-control"
                  >
                    <option value="">Any ward</option>
                    {TOKYO_WARDS.map(w => <option key={w} value={w}>{w}</option>)}
                  </select>
                </div>

                <div className="form-group">
                  <label htmlFor="apartmentType">Room Layout</label>
                  <select
                    id="apartmentType" value={apartmentType}
                    onChange={(e) => setApartmentType(e.target.value)}
                    className="form-control"
                  >
                    <option value="">Any layout</option>
                    {APARTMENT_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                  </select>
                </div>

                <div className="form-group">
                  <label htmlFor="houseType">Building Type</label>
                  <select
                    id="houseType" value={houseType}
                    onChange={(e) => setHouseType(e.target.value)}
                    className="form-control"
                  >
                    <option value="">Any type</option>
                    {HOUSE_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                  </select>
                </div>

                <div className="form-group">
                  <label htmlFor="contractType">Contract Type</label>
                  <select
                    id="contractType" value={contractType}
                    onChange={(e) => setContractType(e.target.value)}
                    className="form-control"
                  >
                    <option value="0">普通借家 (Standard lease)</option>
                    <option value="1">定期借家 (Fixed-term lease)</option>
                  </select>
                </div>
              </div>
            )}
          </div>

          {/* ── Prediction method selector ── */}
          <div className="method-selector-section">
            <h3 className="method-selector-title">Prediction method</h3>
            <div className="method-cards">
              <button
                type="button"
                className={`method-card ${predictionMethod === 'standard' ? 'active' : ''}`}
                onClick={() => setPredictionMethod('standard')}
              >
                <span className="method-card__name">Standard</span>
                <span className="method-card__desc">Production regression model</span>
              </button>
              <button
                type="button"
                className={`method-card ${predictionMethod === 'ensemble' ? 'active' : ''}`}
                onClick={() => setPredictionMethod('ensemble')}
              >
                <span className="method-card__name">Ensemble</span>
                <span className="method-card__desc">Combine multiple models</span>
              </button>
              <button
                type="button"
                className={`method-card ${predictionMethod === 'torch' ? 'active' : ''}`}
                onClick={() => setPredictionMethod('torch')}
              >
                <span className="method-card__name">Deep Learning</span>
                <span className="method-card__desc">TabPFN / TabNet / NN</span>
              </button>
            </div>

            {predictionMethod === 'ensemble' && (
              <div className="method-options">
                <label className="method-options-label">Ensemble method</label>
                <div className="method-option-list">
                  {ENSEMBLE_METHODS.map(m => (
                    <label key={m.value} className={`method-option ${ensembleMethod === m.value ? 'active' : ''}`}>
                      <input
                        type="radio" name="ensembleMethod" value={m.value}
                        checked={ensembleMethod === m.value}
                        onChange={() => setEnsembleMethod(m.value)}
                      />
                      <span className="method-option__name">{m.label}</span>
                      <span className="method-option__desc">{m.desc}</span>
                    </label>
                  ))}
                </div>
              </div>
            )}

            {predictionMethod === 'torch' && (
              <div className="method-options">
                <label className="method-options-label">Model</label>
                <div className="method-option-list">
                  {TORCH_MODELS.map(m => (
                    <label key={m.value} className={`method-option ${torchModel === m.value ? 'active' : ''}`}>
                      <input
                        type="radio" name="torchModel" value={m.value}
                        checked={torchModel === m.value}
                        onChange={() => setTorchModel(m.value)}
                      />
                      <span className="method-option__name">{m.label}</span>
                      <span className="method-option__desc">{m.desc}</span>
                    </label>
                  ))}
                </div>
              </div>
            )}
          </div>

          <button
            type="submit"
            className={`btn-success ${isLoading ? 'loading' : ''}`}
            disabled={isLoading}
            aria-busy={isLoading}
          >
            {isLoading ? 'Calculating...' : 'Calculate Rent!'}
          </button>

          {rentPrice !== null && (
            <div className="rentPrice" role="status" aria-live="polite">
              <p>Estimated rent: <strong>{formatPrediction(rentPrice)}</strong></p>
              {resultModelType && (
                <span className="model-tag">{resultModelType}</span>
              )}
            </div>
          )}

          <p className="disclaimer">
            * Estimated from historical data. Actual prices may vary. Limited to 20 calculations per minute.
          </p>
        </form>
      </div>

      {history.length > 0 && (
        <div className="prediction-history">
          <h3>Recent Predictions</h3>
          <div className="history-table-wrapper">
            <table className="history-table">
              <thead>
                <tr>
                  <th>Result</th>
                  <th>Model</th>
                  <th>m²</th>
                  <th>Age (yrs)</th>
                  <th>Floor</th>
                  <th>Walk</th>
                  <th>Key 礼金</th>
                  <th>Dep. 敷金</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h, i) => (
                  <tr key={i}>
                    <td><strong>¥{Math.round(h.prediction * 10000).toLocaleString()}</strong></td>
                    <td><span className="model-tag small">{h.modelType}</span></td>
                    <td>{h.sqrM}</td>
                    <td>{h.yearBuilt}</td>
                    <td>{h.floor === '0' ? 'G' : `${h.floor}F`}</td>
                    <td>{h.ekiWalk}m</td>
                    <td>{h.reiPrice}万</td>
                    <td>{h.shikikin}万</td>
                    <td>{h.timestamp}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <ModelInfo />
    </div>
  );
};

export default Model;
