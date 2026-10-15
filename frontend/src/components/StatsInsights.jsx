import React, { useMemo } from 'react';
import * as d3 from 'd3';

const StatBlock = ({ label, median, q1, q3, unit }) => (
  <div style={{ flex: 1, minWidth: '150px' }}>
    <div style={{ fontSize: '0.8125rem', color: 'var(--text-tertiary)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '6px' }}>
      {label}
    </div>
    <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)' }}>
      {median}{unit}
    </div>
    <div style={{ fontSize: '0.8125rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
      Q1 {q1} · Q3 {q3}
    </div>
    <div style={{ fontSize: '0.8125rem', color: 'var(--text-tertiary)' }}>
      IQR {(q3 - q1).toFixed(1)}
    </div>
  </div>
);

const StatsInsights = ({ data }) => {
  const stats = useMemo(() => {
    if (!data || data.length === 0) return null;

    const sortedRent = data.map(d => d.rent_price).sort(d3.ascending);
    const sortedSqm = data.map(d => d.sqr_m).sort(d3.ascending);
    const sortedWalk = data.map(d => d.eki_walk).sort(d3.ascending);

    const wardCounts = d3.rollup(data, v => v.length, d => d.ku_name);
    const topWard = Array.from(wardCounts).sort((a, b) => b[1] - a[1])[0];

    const typeCounts = d3.rollup(data, v => v.length, d => d.apartment_type);
    const topType = Array.from(typeCounts).sort((a, b) => b[1] - a[1])[0];
    const topTypePct = topType ? ((topType[1] / data.length) * 100).toFixed(0) : null;

    return {
      rent: {
        median: d3.median(sortedRent)?.toFixed(1) ?? '—',
        q1: d3.quantile(sortedRent, 0.25)?.toFixed(1) ?? '—',
        q3: d3.quantile(sortedRent, 0.75)?.toFixed(1) ?? '—',
      },
      sqm: {
        median: d3.median(sortedSqm)?.toFixed(1) ?? '—',
        q1: d3.quantile(sortedSqm, 0.25)?.toFixed(1) ?? '—',
        q3: d3.quantile(sortedSqm, 0.75)?.toFixed(1) ?? '—',
      },
      walk: {
        median: d3.median(sortedWalk)?.toFixed(0) ?? '—',
        q1: d3.quantile(sortedWalk, 0.25)?.toFixed(0) ?? '—',
        q3: d3.quantile(sortedWalk, 0.75)?.toFixed(0) ?? '—',
      },
      topWard,
      topType,
      topTypePct,
    };
  }, [data]);

  if (!stats) return null;

  return (
    <div style={{
      background: 'var(--surface)',
      border: '1px solid var(--border-light)',
      borderRadius: 'var(--radius-xl)',
      padding: 'var(--spacing-lg) var(--spacing-xl)',
      marginBottom: 'var(--spacing-xl)',
      boxShadow: 'var(--shadow-md)',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--spacing-md)' }}>
        <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)' }}>
          Statistical Insights
        </h3>
        <span style={{ fontSize: '0.8125rem', color: 'var(--text-tertiary)' }}>
          {data.length.toLocaleString()} listings
        </span>
      </div>

      <div style={{ display: 'flex', gap: 'var(--spacing-xl)', flexWrap: 'wrap', marginBottom: 'var(--spacing-md)' }}>
        <StatBlock
          label="Rent (万円)"
          median={stats.rent.median}
          q1={stats.rent.q1}
          q3={stats.rent.q3}
          unit=""
        />
        <StatBlock
          label="Size (m²)"
          median={stats.sqm.median}
          q1={stats.sqm.q1}
          q3={stats.sqm.q3}
          unit=""
        />
        <StatBlock
          label="Walk to Station"
          median={stats.walk.median}
          q1={stats.walk.q1}
          q3={stats.walk.q3}
          unit=" min"
        />
      </div>

      <div style={{ display: 'flex', gap: 'var(--spacing-xl)', flexWrap: 'wrap', borderTop: '1px solid var(--border-light)', paddingTop: 'var(--spacing-md)', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
        {stats.topWard && (
          <span>Most common ward: <strong style={{ color: 'var(--text-primary)' }}>{stats.topWard[0]}</strong> ({stats.topWard[1]} listings)</span>
        )}
        {stats.topType && (
          <span>Most common type: <strong style={{ color: 'var(--text-primary)' }}>{stats.topType[0]}</strong> ({stats.topTypePct}%)</span>
        )}
      </div>
    </div>
  );
};

export default StatsInsights;
