import * as d3 from 'd3';

/**
 * Parse CSV data and transform to usable format
 * @param {string} csvText - Raw CSV text
 * @returns {Array} Parsed and transformed data
 */
export const parseTokyoRentData = (csvText) => {
  const parsed = d3.csvParse(csvText);

  return parsed.map(row => ({
    rent_price: parseFloat(row.rent_price),
    address: row.address,
    sqr_m: parseFloat(row.sqr_m),
    rei_price: parseFloat(row.rei_price),
    shikikin: parseFloat(row.shikikin),
    maintenence_price: parseFloat(row.maintenence_price),
    apartment_type: row.apartment_type,
    year_built: parseInt(row.year_built, 10),
    ku_name: row.ku_name,
    floor: parseInt(row.floor, 10),
    house_type: row.house_type,
    eki_walk: parseInt(row.eki_walk, 10),
    // year_built in the dataset is years since construction (0 = new build)
    price_per_sqm: parseFloat(row.rent_price) / parseFloat(row.sqr_m),
  })).filter(d => !isNaN(d.rent_price) && !isNaN(d.sqr_m));
};

/**
 * Calculate average rent by ward (ku)
 * @param {Array} data - Parsed data array
 * @returns {Array} Array of {ku_name, avgRent, count}
 */
export const getAverageRentByWard = (data) => {
  const grouped = d3.group(data, d => d.ku_name);

  return Array.from(grouped, ([ku_name, values]) => ({
    ku_name,
    avgRent: d3.mean(values, d => d.rent_price),
    count: values.length,
    minRent: d3.min(values, d => d.rent_price),
    maxRent: d3.max(values, d => d.rent_price),
  })).sort((a, b) => b.avgRent - a.avgRent);
};

/**
 * Calculate average rent by apartment type
 * @param {Array} data - Parsed data array
 * @returns {Array} Array of {apartment_type, avgRent, count}
 */
export const getAverageRentByType = (data) => {
  const grouped = d3.group(data, d => d.apartment_type);

  return Array.from(grouped, ([apartment_type, values]) => ({
    apartment_type,
    avgRent: d3.mean(values, d => d.rent_price),
    count: values.length,
    avgSize: d3.mean(values, d => d.sqr_m),
  })).sort((a, b) => b.avgRent - a.avgRent);
};

/**
 * Get price distribution bins for histogram
 * @param {Array} data - Parsed data array
 * @param {number} numBins - Number of bins (default: 20)
 * @returns {Array} Binned data
 */
export const getPriceDistribution = (data, numBins = 20) => {
  const prices = data.map(d => d.rent_price);
  const histogram = d3.histogram()
    .domain([0, d3.max(prices)])
    .thresholds(numBins);

  return histogram(prices);
};

/**
 * Get summary statistics
 * @param {Array} data - Parsed data array
 * @returns {Object} Summary statistics
 */
export const getSummaryStats = (data) => {
  const rentPrices = data.map(d => d.rent_price);
  const sizes = data.map(d => d.sqr_m);

  return {
    totalListings: data.length,
    avgRent: d3.mean(rentPrices),
    medianRent: d3.median(rentPrices),
    minRent: d3.min(rentPrices),
    maxRent: d3.max(rentPrices),
    avgSize: d3.mean(sizes),
    medianSize: d3.median(sizes),
    wards: new Set(data.map(d => d.ku_name)).size,
    apartmentTypes: new Set(data.map(d => d.apartment_type)).size,
  };
};

/**
 * Filter data based on criteria
 * @param {Array} data - Parsed data array
 * @param {Object} filters - Filter criteria
 * @returns {Array} Filtered data
 */
export const filterData = (data, filters = {}) => {
  let filtered = [...data];

  if (filters.kuName && filters.kuName.length > 0) {
    filtered = filtered.filter(d => filters.kuName.includes(d.ku_name));
  }

  if (filters.apartmentType && filters.apartmentType.length > 0) {
    filtered = filtered.filter(d => filters.apartmentType.includes(d.apartment_type));
  }

  if (filters.minRent !== undefined) {
    filtered = filtered.filter(d => d.rent_price >= filters.minRent);
  }

  if (filters.maxRent !== undefined) {
    filtered = filtered.filter(d => d.rent_price <= filters.maxRent);
  }

  if (filters.minSize !== undefined) {
    filtered = filtered.filter(d => d.sqr_m >= filters.minSize);
  }

  if (filters.maxSize !== undefined) {
    filtered = filtered.filter(d => d.sqr_m <= filters.maxSize);
  }

  if (filters.maxWalkTime !== undefined) {
    filtered = filtered.filter(d => d.eki_walk <= filters.maxWalkTime);
  }

  return filtered;
};

/**
 * Get correlation between two variables
 * @param {Array} data - Parsed data array
 * @param {string} xField - Field name for x-axis
 * @param {string} yField - Field name for y-axis
 * @returns {number} Pearson correlation coefficient
 */
export const getCorrelation = (data, xField, yField) => {
  const xValues = data.map(d => d[xField]).filter(v => !isNaN(v));
  const yValues = data.map(d => d[yField]).filter(v => !isNaN(v));

  if (xValues.length !== yValues.length || xValues.length === 0) return 0;

  const xMean = d3.mean(xValues);
  const yMean = d3.mean(yValues);

  let numerator = 0;
  let xDenominator = 0;
  let yDenominator = 0;

  for (let i = 0; i < xValues.length; i++) {
    const xDiff = xValues[i] - xMean;
    const yDiff = yValues[i] - yMean;
    numerator += xDiff * yDiff;
    xDenominator += xDiff * xDiff;
    yDenominator += yDiff * yDiff;
  }

  const denominator = Math.sqrt(xDenominator * yDenominator);
  return denominator === 0 ? 0 : numerator / denominator;
};

/**
 * Get unique values for a field
 * @param {Array} data - Parsed data array
 * @param {string} field - Field name
 * @returns {Array} Sorted unique values
 */
export const getUniqueValues = (data, field) => {
  return [...new Set(data.map(d => d[field]))].sort();
};
