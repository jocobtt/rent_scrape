import { useRef, useCallback } from 'react';

/**
 * Hook that rate limits function calls
 * @param {Function} callback - The function to rate limit
 * @param {number} limit - Maximum number of calls
 * @param {number} interval - Time window in milliseconds
 * @returns {Function} Rate-limited function
 */
export const useRateLimit = (callback, limit = 5, interval = 60000) => {
  const callTimesRef = useRef([]);

  const rateLimitedFunction = useCallback(
    (...args) => {
      const now = Date.now();

      // Remove calls outside the time window
      callTimesRef.current = callTimesRef.current.filter(
        (time) => now - time < interval
      );

      // Check if we've exceeded the rate limit
      if (callTimesRef.current.length >= limit) {
        const oldestCall = callTimesRef.current[0];
        const timeUntilNextCall = interval - (now - oldestCall);
        const minutesRemaining = Math.ceil(timeUntilNextCall / 60000);

        throw new Error(
          `Rate limit exceeded. Please try again in ${minutesRemaining} minute${minutesRemaining > 1 ? 's' : ''}.`
        );
      }

      // Record this call
      callTimesRef.current.push(now);

      // Execute the callback
      return callback(...args);
    },
    [callback, limit, interval]
  );

  return rateLimitedFunction;
};

export default useRateLimit;
