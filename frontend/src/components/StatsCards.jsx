import React, { useMemo } from 'react';
import './StatsCards.css';

const StatsCards = ({ data }) => {
  const stats = useMemo(() => {
    if (!data || data.length === 0) {
      return {
        avgRent: 0,
        avgSqm: 0,
        avgPricePerSqm: 0,
        totalListings: 0,
      };
    }

    const avgRent = data.reduce((sum, d) => sum + d.rent_price, 0) / data.length;
    const avgSqm = data.reduce((sum, d) => sum + d.sqr_m, 0) / data.length;
    const avgPricePerSqm = avgSqm > 0 ? avgRent / avgSqm : 0;

    return {
      avgRent: avgRent.toFixed(1),
      avgSqm: avgSqm.toFixed(1),
      avgPricePerSqm: (avgPricePerSqm * 10000).toFixed(0), // Convert to yen per sqm
      totalListings: data.length,
    };
  }, [data]);

  const cards = [
    {
      label: 'Average Rent',
      value: `¥${(stats.avgRent * 10000).toLocaleString()}`,
      subtitle: `${stats.avgRent}万円/month`,
      color: 'primary',
    },
    {
      label: 'Average Size',
      value: `${stats.avgSqm} m²`,
      subtitle: 'Square meters',
      color: 'secondary',
    },
    {
      label: 'Price per m²',
      value: `¥${stats.avgPricePerSqm}`,
      subtitle: 'Per square meter',
      color: 'accent',
    },
    {
      label: 'Total Listings',
      value: stats.totalListings.toLocaleString(),
      subtitle: 'Apartments available',
      color: 'info',
    },
  ];

  return (
    <div className="stats-cards">
      {cards.map((card, index) => (
        <div key={index} className={`stat-card stat-card-${card.color}`}>
          <div className={`stat-dot stat-dot-${card.color}`}></div>
          <div className="stat-content">
            <div className="stat-label">{card.label}</div>
            <div className="stat-value">{card.value}</div>
            <div className="stat-subtitle">{card.subtitle}</div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default StatsCards;
