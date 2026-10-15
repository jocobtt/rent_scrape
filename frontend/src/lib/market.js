// Helpers over GET /data/market: starting values for the guided rent form and how an
// estimate compares with what is typically listed. Rents are in 万円.

// Typical held-out error of the no-deposit / key-money LightGBM, in 万円 (RMSE; a rough guide,
// not a confidence interval). 4.0 with a station to look nearby rents up by, 6.3 without any
// location beyond the ward (CLAUDE.md "Per-m² target and residual lag"; README no-terms table).
export const TYPICAL_ERROR = { withStation: 4.0, wardOnly: 6.3 };

const num = (v) => String(Math.round(v * 10) / 10);

/** Medians for a ward + layout, falling back to the ward, then all of Tokyo. Null without market data. */
export function cellFor(market, ward, layout) {
  if (!market) return null;
  const w = market.wards[ward];
  return w?.layouts?.[layout] ?? (layout ? market.tokyo.layouts[layout] : null) ?? w ?? market.tokyo;
}

/** Form values (strings) a renter starts from; they change only what they care about. */
export function defaultsFor(market, ward, layout, station) {
  const cell = cellFor(market, ward, layout);
  if (!cell) return null;
  return {
    sqrM: num(cell.sqr_m),
    maintenencePrice: String(cell.maintenance),
    yearBuilt: String(Math.round(cell.year_built)),
    floor: String(Math.round(cell.floor)),
    // A station's own walk time is a better start than the ward's
    ekiWalk: String(Math.round(station?.eki_walk ?? cell.eki_walk)),
  };
}

/** Layouts present in this ward, most listed first, as ChipGroup options. */
export function layoutOptions(market, ward) {
  const layouts = market?.wards[ward]?.layouts ?? market?.tokyo.layouts ?? {};
  return Object.entries(layouts)
    .map(([value, cell]) => ({ value, count: cell.count }))
    .sort((a, b) => b.count - a.count);
}

const wardName = (ward) => ward.replace(/区$/, '');

/** Wards and stations matching `query` (substring, so typing 渋 finds 渋谷区 and 渋谷). */
export function searchPlaces(market, query, limit = 8) {
  const q = query.trim();
  if (!market || !q) return [];
  const wards = Object.entries(market.wards)
    .filter(([name]) => name.includes(q) || wardName(name).includes(q))
    .map(([name, w]) => ({ kind: 'ward', key: `w-${name}`, label: name, ward: name, count: w.count }));
  const stations = market.stations
    .filter((s) => s.name.includes(q))
    // Exact and prefix matches first, then the busiest
    .sort((a, b) => (b.name.startsWith(q) - a.name.startsWith(q)) || (b.count - a.count))
    .map((s) => ({ kind: 'station', key: `s-${s.name}-${s.line}`, label: `${s.name}駅`, station: s, ward: s.ward, count: s.count }));
  return [...wards, ...stations].slice(0, limit);
}

/**
 * How an estimate compares with the typical listing of the same layout in that ward (per m²,
 * so a bigger-than-usual flat isn't called expensive for being big).
 * @returns {{pct: number, typicalRent: number, count: number}|null}
 */
export function compareToTypical(market, ward, layout, sqrM, rent) {
  const cell = market?.wards[ward]?.layouts?.[layout];
  if (!cell || !(sqrM > 0)) return null;
  const typicalPerSqm = cell.rent_per_sqm;
  return {
    pct: Math.round(((rent / sqrM) / typicalPerSqm - 1) * 100),
    typicalRent: typicalPerSqm * sqrM,
    count: cell.count,
  };
}

/**
 * What the deposit usually is for this flat. Deposit equals one month's rent in 83% of 港区 2LDK
 * listings (and the models lean on it heavily), so the typical rent for the size is a good guide.
 * @returns {number|null} 万円
 */
export function typicalDeposit(market, ward, layout, sqrM) {
  const typical = compareToTypical(market, ward, layout, sqrM, 0);
  return typical ? Math.round(typical.typicalRent) : null;
}

/** [low, high] around a prediction, never below zero. */
export function estimateRange(rent, hasStation) {
  const err = hasStation ? TYPICAL_ERROR.withStation : TYPICAL_ERROR.wardOnly;
  return [Math.max(0, rent - err), rent + err];
}
