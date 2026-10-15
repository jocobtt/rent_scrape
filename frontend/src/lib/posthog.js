// posthog-js is large and not needed for first paint, so it is imported on demand.
// Events fired before it has loaded are queued on the promise.
let clientPromise = null;
let client = null;

/**
 * Initialize PostHog analytics
 * Call this once when your app loads
 * @returns {Promise|null} resolves to the client; null when no API key is configured
 */
export const initPostHog = () => {
  // Get PostHog configuration from environment variables
  const posthogKey = import.meta.env.VITE_POSTHOG_KEY;
  const posthogHost = import.meta.env.VITE_POSTHOG_HOST || 'https://app.posthog.com';

  if (!posthogKey) {
    console.warn('PostHog API key not found. Analytics will be disabled.');
    return null;
  }
  if (clientPromise) return clientPromise;

  clientPromise = import('posthog-js').then(({ default: posthog }) => {
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
      loaded: (_ph) => {
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

    client = posthog;
    return posthog;
  });

  return clientPromise;
};

// Run `fn(posthog)` once the client has loaded; a no-op when analytics are disabled
const whenLoaded = (fn) => {
  clientPromise?.then((posthog) => {
    if (posthog.__loaded) fn(posthog);
  });
};

/**
 * Track a custom event
 * @param {string} eventName - Name of the event
 * @param {object} properties - Event properties/metadata
 */
export const trackEvent = (eventName, properties = {}) => {
  whenLoaded((posthog) => posthog.capture(eventName, properties));
};

/**
 * Identify a user
 * @param {string} userId - Unique user identifier
 * @param {object} properties - User properties
 */
export const identifyUser = (userId, properties = {}) => {
  whenLoaded((posthog) => posthog.identify(userId, properties));
};

/**
 * Reset user identity (e.g., on logout)
 */
export const resetUser = () => {
  whenLoaded((posthog) => posthog.reset());
};

/**
 * Set user properties
 * @param {object} properties - Properties to set
 */
export const setUserProperties = (properties) => {
  whenLoaded((posthog) => posthog.setPersonProperties(properties));
};

/**
 * Track page view manually (useful for SPAs)
 * @param {string} pageName - Name/path of the page
 */
export const trackPageView = (pageName) => {
  whenLoaded((posthog) =>
    posthog.capture('$pageview', {
      $current_url: window.location.href,
      page_name: pageName,
    }),
  );
};

/**
 * Create a feature flag check
 * Synchronous, so it returns `defaultValue` until the client has loaded.
 * @param {string} flagKey - Feature flag key
 * @param {any} defaultValue - Default value if flag doesn't exist
 * @returns {any} Flag value
 */
export const getFeatureFlag = (flagKey, defaultValue = false) => {
  if (client?.__loaded) {
    return client.getFeatureFlag(flagKey) ?? defaultValue;
  }
  return defaultValue;
};

/**
 * Check if a feature flag is enabled (false until the client has loaded)
 * @param {string} flagKey - Feature flag key
 * @returns {boolean} True if enabled
 */
export const isFeatureFlagEnabled = (flagKey) => {
  if (client?.__loaded) {
    return client.isFeatureEnabled(flagKey);
  }
  return false;
};
