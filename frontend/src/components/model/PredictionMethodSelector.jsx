import React from 'react';

const ENSEMBLE_METHODS = [
  { value: 'weighted_average', label: 'Weighted Average', desc: 'Optimally weighted combination of all models' },
  { value: 'simple_average',   label: 'Simple Average',   desc: 'Equal-weight average of all models' },
  { value: 'stacking',         label: 'Stacking',         desc: 'Meta-model trained to combine predictions' },
  { value: 'best_model',       label: 'Best Model',       desc: 'Routes to the highest-performing model' },
];

const TORCH_MODELS = [
  { value: 'tokyo_rent_tabpfn', label: 'TabPFN',    desc: 'Pretrained transformer for tabular data' },
  { value: 'tokyo_rent_tabnet', label: 'TabNet',    desc: 'Interpretable attention-based deep learning' },
  { value: 'tokyo_rent_torch',  label: 'Custom NN', desc: 'Custom PyTorch neural network' },
];

const RadioOptions = ({ name, label, items, selected, onSelect }) => (
  <div className="method-options">
    <label className="method-options-label">{label}</label>
    <div className="method-option-list">
      {items.map((m) => (
        <label key={m.value} className={`method-option ${selected === m.value ? 'active' : ''}`}>
          <input
            type="radio" name={name} value={m.value}
            checked={selected === m.value}
            onChange={() => onSelect(m.value)}
          />
          <span className="method-option__name">{m.label}</span>
          <span className="method-option__desc">{m.desc}</span>
        </label>
      ))}
    </div>
  </div>
);

const MethodCard = ({ name, desc, active, disabled, onClick }) => (
  <button
    type="button"
    className={`method-card ${active ? 'active' : ''}`}
    onClick={onClick}
    disabled={disabled}
  >
    <span className="method-card__name">{name}</span>
    <span className="method-card__desc">{desc}</span>
  </button>
);

/**
 * Standard / Ensemble / Deep Learning picker. `available.*` is null while checking,
 * false when that family has not been trained (its card is then disabled).
 */
const PredictionMethodSelector = ({
  method, onMethodChange, ensembleMethod, onEnsembleMethodChange,
  torchModel, onTorchModelChange, available,
}) => (
  <div className="method-selector-section">
    <h3 className="method-selector-title">Prediction method</h3>
    <div className="method-cards">
      <MethodCard
        name="Standard" desc="Production regression model"
        active={method === 'standard'} onClick={() => onMethodChange('standard')}
      />
      <MethodCard
        name="Ensemble"
        desc={available.ensemble === false ? 'Not trained yet' : 'Combine multiple models'}
        active={method === 'ensemble'} disabled={available.ensemble === false}
        onClick={() => onMethodChange('ensemble')}
      />
      <MethodCard
        name="Deep Learning"
        desc={available.torch === false ? 'Not trained yet' : 'TabPFN / TabNet / NN'}
        active={method === 'torch'} disabled={available.torch === false}
        onClick={() => onMethodChange('torch')}
      />
    </div>

    {method === 'ensemble' && (
      <RadioOptions
        name="ensembleMethod" label="Ensemble method" items={ENSEMBLE_METHODS}
        selected={ensembleMethod} onSelect={onEnsembleMethodChange}
      />
    )}
    {method === 'torch' && (
      <RadioOptions
        name="torchModel" label="Model" items={TORCH_MODELS}
        selected={torchModel} onSelect={onTorchModelChange}
      />
    )}
  </div>
);

export default PredictionMethodSelector;
