// Shared by the Predict Rent (/model) and Compare (/compare) pages: option lists,
// validation and the request body for the prediction endpoints.

export const FLOOR_OPTIONS = Array.from({ length: 61 }, (_, i) => i);

export const TOKYO_WARDS = [
  '千代田区', '中央区', '港区', '新宿区', '文京区', '台東区',
  '墨田区', '江東区', '品川区', '目黒区', '大田区', '世田谷区',
  '渋谷区', '中野区', '杉並区', '豊島区', '北区', '荒川区',
  '板橋区', '練馬区', '足立区', '葛飾区', '江戸川区',
];

export const APARTMENT_TYPES = [
  'ワンルーム', '1K', '1DK', '1LDK',
  '2K', '2DK', '2LDK',
  '3K', '3DK', '3LDK', '4K以上',
];

export const HOUSE_TYPES = [
  '賃貸マンション', '賃貸アパート', '賃貸一戸建て', '賃貸テラスハウス',
];

// Form values are kept as strings (what the inputs hold); buildRentInput converts them
export const EMPTY_VALUES = {
  sqrM: '', reiPrice: '', shikikin: '', maintenencePrice: '',
  yearBuilt: '', floor: '', ekiWalk: '',
  kuName: '', apartmentType: '', houseType: '',
  address: '', nearestStation: '', nearestLine: '',
};

export const floorLabel = (n) => (n === 0 ? 'Ground Floor' : `${n}F`);

/**
 * @param values        form values (strings)
 * @param termsRequired whether deposit / key money must be filled in (every method but Standard)
 * @returns {Object} field name -> message; empty when valid
 */
export function validateRentForm(values, { termsRequired }) {
  const { sqrM, reiPrice, shikikin, maintenencePrice, yearBuilt, floor, ekiWalk } = values;
  const errors = {};
  if (!sqrM || sqrM < 5 || sqrM > 500)
    errors.sqrM = 'Enter a valid size (5–500 sqm)';
  if ((termsRequired || reiPrice !== '') && (reiPrice === '' || reiPrice < 0 || reiPrice > 100))
    errors.reiPrice = 'Enter key money (0–100 万円)';
  if ((termsRequired || shikikin !== '') && (shikikin === '' || shikikin < 0 || shikikin > 100))
    errors.shikikin = 'Enter security deposit (0–100 万円)';
  if (maintenencePrice === '' || maintenencePrice < 0 || maintenencePrice > 50)
    errors.maintenencePrice = 'Enter maintenance fee (0–50 万円)';
  if (yearBuilt === '' || yearBuilt < 0 || yearBuilt > 100)
    errors.yearBuilt = 'Enter building age in years (0–100)';
  if (floor === '')
    errors.floor = 'Select a floor';
  if (!ekiWalk || ekiWalk < 0 || ekiWalk > 120)
    errors.ekiWalk = 'Enter walk time to station (0–120 min)';
  return errors;
}

/** Request body for /predict, /ensemble/* and /torch/*; blank optional fields are left out. */
export function buildRentInput(values) {
  const { address, nearestStation, nearestLine } = values;
  return {
    sqr_m: parseFloat(values.sqrM),
    // Omitted when blank: the API then uses the model trained without deposit / key money
    ...(values.reiPrice !== '' && { rei_price: parseFloat(values.reiPrice) }),
    ...(values.shikikin !== '' && { shikikin: parseFloat(values.shikikin) }),
    maintenence_price: parseFloat(values.maintenencePrice),
    year_built: parseFloat(values.yearBuilt),
    floor: parseFloat(values.floor),
    eki_walk: parseFloat(values.ekiWalk),
    ...(values.kuName && { ku_name: values.kuName }),
    ...(values.apartmentType && { apartment_type: values.apartmentType }),
    ...(values.houseType && { house_type: values.houseType }),
    // Used by the no-deposit / key-money model to look at nearby rents
    ...(address.trim() && { address: address.trim() }),
    ...(nearestStation.trim() && { nearest_station: nearestStation.trim() }),
    ...(nearestLine.trim() && { nearest_line: nearestLine.trim() }),
  };
}

export const formatYen = (value) => `¥${Math.round(value * 10000).toLocaleString('ja-JP')}`;
export const formatMan = (value) => `${value.toFixed(2)} 万円`;
