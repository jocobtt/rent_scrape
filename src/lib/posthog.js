import posthog from 'posthog-js';

/**
 * Initialize PostHog analytics
 * Call this once when your app loads
 */
export const initPostHog = () => {
  // Get PostHog configuration from environment variables
  const posthogKey = import.meta.env.VITE_POSTHOG_KEY;
  const posthogHost = import.meta.env.VITE_POSTHOG_HOST || 'https://app.posthog.com';

  if (!posthogKey) {
    console.warn('PostHog API key not found. Analytics will be disabled.');
    return null;
  }

  // Initialize PostHog
  posthog.init(posthogKey, {
    api_host: posthogHost,

    // Automatically capture page views
    capture_pageview: false,

    // Automatically capture page leaves
    capture_pageleave: true,

    // Capture performance metrics
    capture_performance: true,

    // Track sessions for better user flow analysis
    session_recording: {
      enabled: true,
      recordCrossOriginIframes: false,
    },

    // Disable in development for cleaner logs
    loaded: (ph) => {
      if (import.meta.env.DEV) {
        console.log('PostHog initialized in development mode');
      }
    },

    // Respect user privacy preferences
    respect_dnt: true,

    // Persist user identity across sessions
    persistence: 'localStorage',

    // Advanced configuration
    autocapture: {
      // Capture clicks, form submits, and other interactions
      capture_copied_text: true,
      css_selector_allowlist: ['.btn', 'button', 'a', 'input[type="submit"]'],
    },
  });

  return posthog;
};

/**
 * Track a custom event
 * @param {string} eventName - Name of the event
 * @param {object} properties - Event properties/metadata
 */
export const trackEvent = (eventName, properties = {}) => {
  if (posthog.__loaded) {
    posthog.capture(eventName, properties);
  }
};

/**
 * Identify a user
 * @param {string} userId - Unique user identifier
 * @param {object} properties - User properties
 */
export const identifyUser = (userId, properties = {}) => {
  if (posthog.__loaded) {
    posthog.identify(userId, properties);
  }
};

/**
 * Reset user identity (e.g., on logout)
 */
export const resetUser = () => {
  if (posthog.__loaded) {
    posthog.reset();
  }
};

/**
 * Set user properties
 * @param {object} properties - Properties to set
 */
export const setUserProperties = (properties) => {
  if (posthog.__loaded) {
    posthog.setPersonProperties(properties);
  }
};

/**
 * Track page view manually (useful for SPAs)
 * @param {string} pageName - Name/path of the page
 */
export const trackPageView = (pageName) => {
  if (posthog.__loaded) {
    posthog.capture('$pageview', {
      $current_url: window.location.href,
      page_name: pageName,
    });
  }
};

/**
 * Create a feature flag check
 * @param {string} flagKey - Feature flag key
 * @param {any} defaultValue - Default value if flag doesn't exist
 * @returns {any} Flag value
 */
export const getFeatureFlag = (flagKey, defaultValue = false) => {
  if (posthog.__loaded) {
    return posthog.getFeatureFlag(flagKey) ?? defaultValue;
  }
  return defaultValue;
};

/**
 * Check if a feature flag is enabled
 * @param {string} flagKey - Feature flag key
 * @returns {boolean} True if enabled
 */
export const isFeatureFlagEnabled = (flagKey) => {
  if (posthog.__loaded) {
    return posthog.isFeatureEnabled(flagKey);
  }
  return false;
};

export default posthog;
