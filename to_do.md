- Improve our main dashboard
  - Need to determine if we want the emojis or not? - done
  - Beter filtering?
  - Deal with data location better, maybe have it served differently for vercel so it isn't stored locally? 
  - Fix year reading, built in year not how many years it is
  - size for rent vs square meters needs to be improved 
  - heatmap needs to be interlaid with map of tokyo probably - done but need to test
  - bar charts need to have some level of click and view data etc. - confirm 
  - building age distribution bar chart needs to be adjusted 
  - Make sure our filter works - can maybe be improved more 
  - Improve max walk time to show 30+ for max - done (test)
- Remove or adjust the visualize rent part - done
- mlflow/model viz section - test

## Performance & Architecture
- [ ] Move Dashboard data/filters to Context API to reduce prop drilling
- [ ] Wrap all D3 visualization components with React.memo()
- [ ] Add useMemo to filter calculations in Dashboard
- [ ] Cache CSV data in localStorage to avoid re-fetching
- [ ] Replace full D3 library with modular imports (reduce bundle ~70%)

## Code Quality
- [ ] Configure ESLint with React/Hooks rules
- [ ] Add Prettier and format all files
- [ ] Create src/lib/constants.js for hardcoded values
- [ ] Consolidate Model.jsx form state into useReducer
- [ ] Add PropTypes to all components OR migrate to TypeScript
- [ ] Create reusable useD3Chart hook to reduce duplication

## Testing
- [ ] Add unit tests for dataUtils.js functions
- [ ] Add component tests for DataFilters and ModelInfo
- [ ] Setup testing library (@testing-library/react)

## Accessibility
- [ ] Add keyboard navigation to InteractiveScatterPlot
- [ ] Add <title> and <desc> to all SVG charts
- [ ] Add ARIA live regions for filter updates
- [ ] Audit with axe-core DevTools

## Developer Experience
- [ ] Setup pre-commit linting
- [ ] Add .env.example for environment variables
- [ ] Configure Axios baseURL via environment variable
- ## Design & UX
- [ ] Implement dark mode with localStorage persistence
- [ ] Add loading skeletons for all charts (replace spinners)
- [ ] Create empty state designs for no-results scenarios
- [ ] Add chart export functionality (PNG/SVG download)
- [ ] Implement fullscreen mode for charts
- [ ] Add zoom and pan to scatter plot
- [ ] Create brushing/linking between charts (coordinated views)

## Configuration & Infrastructure
- [ ] Create src/config/ with environment-based configuration
- [ ] Setup .env.development, .env.staging, .env.production
- [ ] Add feature flags system (PostHog or custom)
- [ ] Configure Vite bundle analyzer
- [ ] Add gzip/brotli compression plugin
- [ ] Setup code splitting by route

## New Features - High Value
- [ ] Add rent trend prediction (show 6-month forecast)
- [ ] Implement save/load filter configurations with shareable URLs
- [ ] Create comparison mode (select 2+ wards, side-by-side comparison)
- [ ] Add confidence intervals to ML predictions
- [ ] Show feature importance visualization (what drives rent price?)
- [ ] Implement "similar apartments" finder after prediction

## Data Management
- [ ] Add IndexedDB caching for CSV data (offline support)
- [ ] Implement data export (filtered results to CSV/JSON)
- [ ] Create data versioning system (notify on new data)
- [ ] Add outlier detection highlighting

## Analytics & Insights
- [ ] Build correlation matrix heatmap
- [ ] Add statistical insights panel (quartiles, MAD, skewness)
- [ ] Create rent affordability calculator (salary → recommended price)
- [ ] Implement price per square meter heatmap

## Mobile & PWA
- [ ] Add PWA manifest.json (make installable)
- [ ] Implement touch gestures for mobile (pinch zoom, swipe)
- [ ] Create mobile-optimized filter bottom sheet
- [ ] Add native share API integration

## Developer Experience
- [ ] Setup Storybook for component documentation
- [ ] Add Playwright E2E tests for critical flows
- [ ] Create architecture decision records (ADRs)
