# 🚀 Quick Start Guide - Tokyo Rent Predictor

## Prerequisites

- Node.js 16+ installed
- PostHog account (optional, for analytics)

## Installation

```bash
# Install dependencies
npm install
```

## Configuration

### 1. Set Up Environment Variables

```bash
# Copy the example env file
cp .env.example .env
```

### 2. Add Your PostHog API Key (Optional)

Edit `.env` and add your PostHog key:

```env
VITE_POSTHOG_KEY=phc_your_api_key_here
```

Get your API key from: [https://app.posthog.com/project/settings](https://app.posthog.com/project/settings)

> **Note**: The app works without PostHog, but you won't have analytics.

## Development

```bash
# Start development server
npm run dev

# App will open at http://localhost:5173
```

## Build for Production

```bash
# Create production build
npm run build

# Preview production build
npm run preview
```

## Features

### ✅ Current Features

- **Rent Calculation**: Predict Tokyo apartment rent based on parameters
- **Data Visualization**: Interactive D3.js charts
- **Rate Limiting**: 5 API calls per minute (client-side)
- **Form Validation**: Real-time field validation
- **Toast Notifications**: User feedback for actions
- **Analytics**: Full PostHog integration
- **Responsive Design**: Mobile-first approach
- **Accessibility**: WCAG compliant

### 📊 Tracked Events

The app automatically tracks:
- Page views
- Navigation clicks
- Rent calculations (attempts, success, errors)
- Form validation errors
- Mobile menu interactions

## Project Structure

```
src/
├── components/          # Reusable components
│   ├── NavBar.jsx      # Navigation bar
│   ├── Footer.jsx      # Footer
│   ├── Toast.jsx       # Toast notifications
│   └── *.jsx           # D3 chart components
├── context/            # React contexts
│   ├── ToastContext.jsx    # Toast state
│   └── PostHogContext.jsx  # Analytics provider
├── hooks/              # Custom React hooks
│   ├── useDebounce.js      # Debounce values
│   ├── useRateLimit.js     # Rate limiting
│   └── useResizeObserver.js # Responsive charts
├── lib/                # Utilities
│   └── posthog.js      # Analytics configuration
├── pages/              # Route pages
│   ├── Home.jsx        # Landing page
│   ├── Model.jsx       # Rent calculator
│   ├── Viz.jsx         # Visualizations
│   └── *.jsx           # Other pages
└── style.css           # Global styles
```

## Key Technologies

- **React 18**: UI framework
- **Vite**: Build tool
- **D3.js**: Data visualizations
- **React Router**: Client-side routing
- **PostHog**: Product analytics
- **Axios**: HTTP client

## Development Tips

### Hot Module Replacement

Vite provides instant HMR - changes appear immediately without full reload.

### PostHog in Development

PostHog tracks events in development. Check console for confirmation.

### Rate Limiting

Rate limit is 5 calls/minute. Resets after 60 seconds.

### Custom Events

Track custom events anywhere:

```javascript
import { trackEvent } from '../lib/posthog';

trackEvent('custom_event', {
  property1: 'value',
  property2: 123
});
```

## Common Commands

```bash
# Development
npm run dev           # Start dev server
npm run build         # Production build
npm run preview       # Preview production build
npm run lint          # Run ESLint

# Dependencies
npm install           # Install packages
npm update            # Update packages
npm outdated          # Check for updates
```

## Troubleshooting

### PostHog Not Working

1. Check `.env` file exists with correct key
2. Look for console errors
3. Verify network requests in DevTools
4. Wait 1-2 minutes for events to appear

### Build Errors

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Port Already in Use

```bash
# Vite will automatically use next available port
# Or specify a port:
npm run dev -- --port 3000
```

## Next Steps

1. **Get PostHog Key**: [Sign up here](https://app.posthog.com/signup)
2. **Configure API**: Set up your backend API endpoint
3. **Deploy**: Deploy to Vercel, Netlify, or your platform
4. **Monitor**: Check PostHog dashboard for insights

## Documentation

- [PostHog Setup](./POSTHOG_SETUP.md) - Detailed analytics guide
- [Frontend Improvements](./FRONTEND_IMPROVEMENTS.md) - All features and improvements
- [CLAUDE.md](./CLAUDE.md) - Architecture and development guide

## Support

For issues or questions:
- Check documentation files
- Review PostHog logs
- Check browser console for errors

## License

[Your License Here]

---

**Happy Coding! 🎉**
