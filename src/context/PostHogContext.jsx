import React, { createContext, useContext, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { initPostHog, trackPageView } from '../lib/posthog';

const PostHogContext = createContext(null);

export const usePostHog = () => {
  return useContext(PostHogContext);
};

export const PostHogProvider = ({ children }) => {
  const location = useLocation();

  useEffect(() => {
    // Initialize PostHog on mount
    const posthog = initPostHog();

    return () => {
      // Cleanup on unmount (optional)
      if (posthog) {
        // PostHog cleanup if needed
      }
    };
  }, []);

  useEffect(() => {
    // Track page views on route change
    trackPageView(location.pathname);
  }, [location]);

  return (
    <PostHogContext.Provider value={null}>
      {children}
    </PostHogContext.Provider>
  );
};
