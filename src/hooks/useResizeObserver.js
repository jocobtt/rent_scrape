import { useState, useEffect, useRef } from 'react';

export const useResizeObserver = () => {
  const resizeRef = useRef();
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const observeTarget = resizeRef.current;

    const observer = new ResizeObserver(entries => {
      entries.forEach(entry => {
        setDimensions({
          width: entry.contentRect.width,
          height: entry.contentRect.height
        });
      });
    });

    if (observeTarget) {
      // Set initial dimensions
      const rect = observeTarget.getBoundingClientRect();
      setDimensions({
        width: rect.width,
        height: rect.height
      });

      observer.observe(observeTarget);
    }

    return () => {
      if (observeTarget) {
        observer.unobserve(observeTarget);
      }
    };
  }, []);

  return { resizeRef, dimensions };
};

export default useResizeObserver;
