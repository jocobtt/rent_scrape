import React from 'react';
import { FLOOR_OPTIONS, floorLabel } from '../lib/rentForm';
import '../styles/form.css';

// Label + control + error / hint, shared by the rent forms. `error` is the message or undefined.

const Label = ({ htmlFor, children, required }) => (
  <label htmlFor={htmlFor}>
    {children}
    {required && <> <span className="required">*</span></>}
  </label>
);

const Feedback = ({ error, hint }) => (
  <>
    {error && <span className="error-message" role="alert">{error}</span>}
    {hint && <span className="field-hint">{hint}</span>}
  </>
);

export const NumberField = ({ id, label, required, error, hint, ...inputProps }) => (
  <div className="form-group">
    <Label htmlFor={id} required={required}>{label}</Label>
    <input
      type="number" id={id}
      className={`form-control ${error ? 'error' : ''}`}
      aria-required={required} aria-invalid={error ? 'true' : 'false'}
      {...inputProps}
    />
    <Feedback error={error} hint={hint} />
  </div>
);

export const TextField = ({ id, label, hint, ...inputProps }) => (
  <div className="form-group">
    <Label htmlFor={id}>{label}</Label>
    <input type="text" id={id} className="form-control" {...inputProps} />
    <Feedback hint={hint} />
  </div>
);

export const SelectField = ({ id, label, required, error, children, ...selectProps }) => (
  <div className="form-group">
    <Label htmlFor={id} required={required}>{label}</Label>
    <select
      id={id}
      className={`form-control ${error ? 'error' : ''}`}
      aria-required={required} aria-invalid={error ? 'true' : 'false'}
      {...selectProps}
    >
      {children}
    </select>
    <Feedback error={error} />
  </div>
);

export const FloorSelect = ({ id, ...props }) => (
  <SelectField id={id} label="Floor:" {...props}>
    <option value="">Select floor</option>
    {FLOOR_OPTIONS.map((n) => <option key={n} value={n}>{floorLabel(n)}</option>)}
  </SelectField>
);
