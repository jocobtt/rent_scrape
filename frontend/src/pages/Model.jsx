import React, { useState } from 'react';
import { useToast } from '../context/ToastContext';
import useRateLimit from '../hooks/useRateLimit';
import useRentForm from '../hooks/useRentForm';
import useModelAvailability from '../hooks/useModelAvailability';
import usePredictionHistory from '../hooks/usePredictionHistory';
import { trackEvent } from '../lib/posthog';
import { buildRentInput, formatYen, formatMan } from '../lib/rentForm';
import api from '../lib/api';
import RentFormFields from '../components/RentFormFields';
import OptionalDetails from '../components/model/OptionalDetails';
import PredictionMethodSelector from '../components/model/PredictionMethodSelector';
import PredictionHistory from '../components/model/PredictionHistory';
import ModelInfo from '../components/ModelInfo';
import './Model.css';

// POST the prediction for the chosen method; resolves to the response body
const requestPrediction = async (method, input, { ensembleMethod, torchModel }) => {
  if (method === 'ensemble') {
    const res = await api.post('/ensemble/predict', { input_data: input, method: ensembleMethod }, { timeout: 15000 });
    return res.data;
  }
  if (method === 'torch') {
    const res = await api.post('/torch/predict', { input_data: input, model_name: torchModel }, { timeout: 20000 });
    return res.data;
  }
  const res = await api.post('/predict', input, { timeout: 10000 });
  return res.data;
};

const Model = () => {
  const { values, errors, onChange, loadValues, setErrors, validate } = useRentForm();
  const { history, addEntry, clear: clearHistory } = usePredictionHistory();
  const availability = useModelAvailability();
  const { showSuccess, showError, showWarning } = useToast();

  // Model selection
  const [predictionMethod, setPredictionMethod] = useState('standard');
  const [ensembleMethod, setEnsembleMethod] = useState('weighted_average');
  const [torchModel, setTorchModel] = useState('tokyo_rent_tabpfn');

  // Result
  const [result, setResult] = useState(null); // { prediction, modelType, variant }
  const [isLoading, setIsLoading] = useState(false);

  // null while checking; the ensemble / deep-learning models only exist once they have been trained
  const available = {
    ensemble: availability.loading ? null : (availability.ensemble?.ensemble_methods?.length ?? 0) > 0,
    torch: availability.loading ? null : Boolean(availability.torch?.available),
  };

  // Only the Standard method can predict without deposit / key money
  const termsRequired = predictionMethod !== 'standard';

  const fetchPrediction = () =>
    requestPrediction(predictionMethod, buildRentInput(values), { ensembleMethod, torchModel });
  const rateLimitedFetch = useRateLimit(fetchPrediction, 20, 60000);

  const handleSubmit = async (e) => {
    e.preventDefault();

    const { sqrM, yearBuilt, floor, ekiWalk } = values;
    trackEvent('rent_calculation_attempted', {
      sqrM, yearBuilt, floor, ekiWalk,
      prediction_method: predictionMethod,
      ensemble_method: predictionMethod === 'ensemble' ? ensembleMethod : undefined,
      torch_model: predictionMethod === 'torch' ? torchModel : undefined,
    });

    const validationErrors = validate({ termsRequired });
    if (Object.keys(validationErrors).length > 0) {
      showWarning('Please fill in all required fields correctly.');
      trackEvent('rent_calculation_validation_failed', { errors: Object.keys(validationErrors) });
      return;
    }

    setIsLoading(true);
    setResult(null);
    setErrors({});

    const startTime = Date.now();

    try {
      const response = await rateLimitedFetch();
      const modelType = response.model_type || response.model_name || predictionMethod;

      setResult({ prediction: response.prediction, modelType, variant: response.variant || null });
      addEntry({
        prediction: response.prediction,
        modelType,
        sqrM, yearBuilt, floor, ekiWalk,
        reiPrice: values.reiPrice, shikikin: values.shikikin, maintenencePrice: values.maintenencePrice,
        timestamp: new Date().toISOString(),
      });
      showSuccess('Rent calculated successfully!');

      trackEvent('rent_calculation_success', {
        sqrM, yearBuilt, floor, ekiWalk,
        prediction: response.prediction,
        model_type: response.model_type,
        prediction_method: predictionMethod,
        duration_ms: Date.now() - startTime,
      });
    } catch (error) {
      console.error('Error calculating rent:', error);
      const isRateLimit = !error.response && error.message?.startsWith('Rate limit');
      const msg = isRateLimit
        ? error.message
        : error.response?.data?.detail || error.message || 'Failed to calculate rent. Please try again.';
      showError(msg);
      trackEvent('rent_calculation_error', { error: error.message, sqrM, yearBuilt, prediction_method: predictionMethod });
    } finally {
      setIsLoading(false);
    }
  };

  const handleReuse = (h) => {
    loadValues({
      sqrM: h.sqrM, reiPrice: h.reiPrice, shikikin: h.shikikin, maintenencePrice: h.maintenencePrice,
      yearBuilt: h.yearBuilt, floor: h.floor, ekiWalk: h.ekiWalk,
    });
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="model">
      <div className="model__title">
        <h1>Calculate your rent!</h1>
        <h2>Input apartment parameters:</h2>
      </div>

      <div className="model__content">
        <form className="model__form" onSubmit={handleSubmit} noValidate>
          <RentFormFields
            values={values} errors={errors} onChange={onChange} termsRequired={termsRequired}
          />

          <OptionalDetails values={values} onChange={onChange} />

          <PredictionMethodSelector
            method={predictionMethod} onMethodChange={setPredictionMethod}
            ensembleMethod={ensembleMethod} onEnsembleMethodChange={setEnsembleMethod}
            torchModel={torchModel} onTorchModelChange={setTorchModel}
            available={available}
          />

          <button
            type="submit"
            className={`btn-success ${isLoading ? 'loading' : ''}`}
            disabled={isLoading}
            aria-busy={isLoading}
          >
            {isLoading ? 'Calculating...' : 'Calculate Rent!'}
          </button>

          {result && (
            <div className="rentPrice" role="status" aria-live="polite">
              <p>
                Estimated rent:{' '}
                <strong>{formatYen(result.prediction)} / month ({formatMan(result.prediction)})</strong>
              </p>
              {result.modelType && <span className="model-tag">{result.modelType}</span>}
              {result.variant === 'no_terms' && (
                <span className="field-hint">Estimated without deposit / key money.</span>
              )}
            </div>
          )}

          <p className="disclaimer">
            * Estimated from historical data. Actual prices may vary. Limited to 20 calculations per minute.
          </p>
        </form>
      </div>

      <PredictionHistory history={history} onClear={clearHistory} onReuse={handleReuse} />

      <ModelInfo />
    </div>
  );
};

export default Model;
