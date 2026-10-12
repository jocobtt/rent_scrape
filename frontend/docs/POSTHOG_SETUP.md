# PostHog Analytics Integration Guide

This guide explains how to set up and use PostHog analytics in your Tokyo Rent Predictor application.

## 🚀 Quick Start

### 1. Get Your PostHog API Key

1. Sign up for PostHog at [https://app.posthog.com/signup](https://app.posthog.com/signup)
2. Create a new project or use an existing one
3. Go to **Project Settings** → **Project API Key**
4. Copy your API key (starts with `phc_`)

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Add your PostHog API key:

```env
VITE_POSTHOG_KEY=phc_your_actual_api_key_here
VITE_POSTHOG_HOST=https://app.posthog.com
```

**Important**: Never commit your `.env` file to git! It's already in `.gitignore`.

### 3. Start the Development Server

```bash
npm run dev
```

PostHog will automatically initialize and start tracking events!

## 📊 What's Being Tracked

### Automatic Tracking

PostHog automatically captures:

- **Page Views**: Every page visit with URL and timestamp
- **Page Leaves**: When users leave pages
- **Performance Metrics**: Load times, rendering performance
- **Session Recordings**: User interactions (can be disabled)
- **User Sessions**: Session duration and flow
- **Clicks**: All button and link clicks

### Custom Events

#### Navigation Events
- **Event**: `navigation_clicked`
  - **Properties**: destination, label, source (mobile/desktop)
  - **Triggered**: When users click navigation links

- **Event**: `mobile_menu_toggled`
  - **Properties**: isOpen
  - **Triggered**: When mobile menu is opened/closed

#### Rent Calculation Events
- **Event**: `rent_calculation_attempted`
  - **Properties**: sqrM, apartmentType, kuName, yearBuilt, floor
  - **Triggered**: When user submits the rent calculation form

- **Event**: `rent_calculation_validation_failed`
  - **Properties**: errors (array of field names)
  - **Triggered**: When form validation fails

- **Event**: `rent_calculation_success`
  - **Properties**: sqrM, apartmentType, kuName, yearBuilt, floor, result, duration_ms
  - **Triggered**: When rent is successfully calculated

- **Event**: `rent_calculation_error`
  - **Properties**: error, sqrM, apartmentType, kuName
  - **Triggered**: When API call fails

## 🎯 Adding Custom Events

### Track an Event

```javascript
import { trackEvent } from '../lib/posthog';

// Simple event
trackEvent('button_clicked');

// Event with properties
trackEvent('feature_used', {
  feature_name: 'data_visualization',
  user_action: 'filter_applied',
  timestamp: Date.now()
});
```

### Track User Identity

```javascript
import { identifyUser, setUserProperties } from '../lib/posthog';

// Identify user (e.g., after login)
identifyUser('user123', {
  email: 'user@example.com',
  name: 'John Doe',
  plan: 'premium'
});

// Update user properties
setUserProperties({
  last_calculation: Date.now(),
  total_calculations: 15
});
```

### Reset User (on logout)

```javascript
import { resetUser } from '../lib/posthog';

resetUser();
```

## 📈 Viewing Analytics in PostHog

### Access Your Dashboard

1. Go to [https://app.posthog.com](https://app.posthog.com)
2. Select your project
3. Navigate to different sections:

### Key Sections

#### 1. **Insights** - Custom Analytics
- Create custom dashboards
- Track specific events over time
- Analyze conversion funnels
- Compare user cohorts

#### 2. **Session Recordings**
- Watch real user sessions
- See exactly how users interact
- Identify UX issues
- Filter by events or properties

#### 3. **Feature Flags** (Optional)
- A/B test features
- Roll out features gradually
- Target specific user segments

#### 4. **Persons**
- View individual users
- See their event timeline
- Track user properties

## 🔍 Useful Queries

### Most Popular Pages
```
Event: $pageview
Group by: current_url
Time range: Last 30 days
```

### Conversion Funnel
1. Event: `rent_calculation_attempted`
2. Event: `rent_calculation_success`

### Average API Response Time
```
Event: rent_calculation_success
Average of: duration_ms
```

### Error Rate
```
Event: rent_calculation_error
Formula: (errors / attempts) * 100
```

## 🛠️ Advanced Configuration

### Disable in Development

PostHog is enabled in development by default. To disable:

```javascript
// src/lib/posthog.js
if (import.meta.env.DEV) {
  return null; // Don't initialize in development
}
```

### Disable Session Recordings

```javascript
// src/lib/posthog.js
session_recording: {
  enabled: false, // Disable recordings
}
```

### Custom Configuration

Edit `src/lib/posthog.js` to customize:

```javascript
posthog.init(posthogKey, {
  api_host: posthogHost,

  // Disable automatic page view tracking
  capture_pageview: false,

  // Disable performance tracking
  capture_performance: false,

  // Custom autocapture settings
  autocapture: {
    capture_copied_text: false,
    // Add specific elements to track
    css_selector_allowlist: ['.track-me'],
  },
});
```

## 🔐 Privacy & GDPR Compliance

### Respect Do Not Track

PostHog respects the browser's DNT setting by default:

```javascript
respect_dnt: true,
```

### Opt-Out for Users

Add an opt-out mechanism:

```javascript
import posthog from 'posthog-js';

// Opt out
posthog.opt_out_capturing();

// Opt in
posthog.opt_in_capturing();

// Check status
posthog.has_opted_out_capturing();
```

### Delete User Data

From PostHog dashboard:
1. Go to **Persons**
2. Find the user
3. Click **Delete Person**

## 📱 Self-Hosting PostHog

If you want to self-host:

1. Deploy PostHog using Docker/Kubernetes
2. Update your `.env`:

```env
VITE_POSTHOG_HOST=https://your-posthog-instance.com
```

See: [PostHog Self-Hosting Docs](https://posthog.com/docs/self-host)

## 🐛 Troubleshooting

### PostHog Not Loading

**Check Console**: Look for errors in browser console

```javascript
// Verify PostHog is loaded
console.log(posthog.__loaded); // Should be true
```

**Check API Key**: Ensure `.env` has correct key

```bash
echo $VITE_POSTHOG_KEY
```

**Check Network**: Open Network tab, look for requests to PostHog

### Events Not Appearing

- Wait 1-2 minutes for events to process
- Check if PostHog is initialized: `posthog.__loaded`
- Verify event name matches exactly
- Check PostHog dashboard filters

### Development vs Production

Events tracked in development will appear in your PostHog dashboard. Consider:
- Using separate PostHog projects for dev/prod
- Filtering by environment in PostHog
- Disabling PostHog in development

## 📚 Additional Resources

- [PostHog Documentation](https://posthog.com/docs)
- [PostHog React Integration](https://posthog.com/docs/libraries/react)
- [Event Tracking Best Practices](https://posthog.com/docs/integrate/client/js#capturing-events)
- [Privacy Controls](https://posthog.com/docs/privacy)
- [Feature Flags Guide](https://posthog.com/docs/user-guides/feature-flags)

## 🎓 Event Naming Best Practices

1. **Be Descriptive**: `rent_calculation_success` vs `success`
2. **Use Snake Case**: `form_submitted` vs `FormSubmitted`
3. **Include Context**: `navigation_clicked` not just `clicked`
4. **Group Related Events**:
   - `rent_calculation_attempted`
   - `rent_calculation_success`
   - `rent_calculation_error`

## ✅ Event Properties Best Practices

1. **Include Relevant Context**:
   ```javascript
   trackEvent('rent_calculation_success', {
     sqrM: 80,
     apartmentType: '1DK',
     result: '¥120,000',
     duration_ms: 234
   });
   ```

2. **Use Consistent Property Names**:
   - `user_id` not `userId` or `userID`
   - `timestamp` not `time` or `date`

3. **Avoid PII**: Don't send emails, names, addresses unless necessary

## 📊 Sample Dashboard Setup

Create a dashboard with:

1. **Total Users** (Last 7 days)
2. **Page Views by Page** (Chart)
3. **Rent Calculations** (Success rate)
4. **Average API Response Time** (Line chart)
5. **Top Navigation Clicks** (Table)
6. **Error Rate** (Percentage)

## 🚨 Important Notes

- PostHog is free for up to 1M events/month
- Session recordings count towards event quota
- Consider upgrading for higher volumes
- Set up alerts for anomalies
- Review privacy policies regularly
