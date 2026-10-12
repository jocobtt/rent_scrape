import { useReducer, useCallback } from 'react';
import { EMPTY_VALUES, validateRentForm } from '../lib/rentForm';

function reducer(state, action) {
  switch (action.type) {
    case 'field':
      return { ...state, values: { ...state.values, [action.name]: action.value } };
    case 'load':
      return { ...state, values: { ...state.values, ...action.values } };
    case 'errors':
      return { ...state, errors: action.errors };
    default:
      return state;
  }
}

/** Form state for the rent forms: field values plus validation errors. */
export default function useRentForm() {
  const [{ values, errors }, dispatch] = useReducer(reducer, { values: EMPTY_VALUES, errors: {} });

  // onChange handler factory: <input onChange={onChange('sqrM')} />
  const onChange = useCallback(
    (name) => (e) => dispatch({ type: 'field', name, value: e.target.value }),
    [],
  );
  const setField = useCallback((name, value) => dispatch({ type: 'field', name, value }), []);
  const loadValues = useCallback((v) => dispatch({ type: 'load', values: v }), []);
  const setErrors = useCallback((e) => dispatch({ type: 'errors', errors: e }), []);

  // Stores and returns the errors so the caller can branch on them
  const validate = useCallback((options) => {
    const next = validateRentForm(values, options);
    dispatch({ type: 'errors', errors: next });
    return next;
  }, [values]);

  return { values, errors, onChange, setField, loadValues, setErrors, validate };
}
