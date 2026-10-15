import { useEffect, useState } from 'react';

const KEY = 'rentHistory';
const MAX_ENTRIES = 5;

/** The last few predictions, kept in localStorage. */
export default function usePredictionHistory() {
  const [history, setHistory] = useState(() => {
    try { return JSON.parse(localStorage.getItem(KEY) || '[]'); }
    catch { return []; }
  });

  useEffect(() => {
    try { localStorage.setItem(KEY, JSON.stringify(history)); }
    catch { /* storage unavailable (private mode): history just isn't persisted */ }
  }, [history]);

  const addEntry = (entry) => setHistory((prev) => [entry, ...prev].slice(0, MAX_ENTRIES));
  const clear = () => setHistory([]);

  return { history, addEntry, clear };
}
