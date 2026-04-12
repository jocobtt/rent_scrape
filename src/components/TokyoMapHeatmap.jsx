import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { getAverageRentByWard } from '../lib/dataUtils';
import './TokyoMapHeatmap.css';

const TokyoMapHeatmap = ({ data, width = 800, height = 700 }) => {
  const svgRef = useRef(null);
  const [tokyoGeoJSON, setTokyoGeoJSON] = useState(null);
  const [loading, setLoading] = useState(true);

  // Load Tokyo GeoJSON data
  useEffect(() => {
    // Fetch GeoJSON for Tokyo's 23 wards
    // Using a public source for Tokyo ward boundaries
    fetch('https://raw.githubusercontent.com/dataofjapan/land/master/tokyo.geojson')
      .then(response => response.json())
      .then(geoData => {
        setTokyoGeoJSON(geoData);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error loading Tokyo GeoJSON:', error);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    if (!data || data.length === 0 || !svgRef.current || !tokyoGeoJSON) return;

    // Clear previous chart
    d3.select(svgRef.current).selectAll('*').remove();

    const margin = { top: 50, right: 50, bottom: 50, left: 50 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    // Get ward statistics
    const wardStats = getAverageRentByWard(data);
    const wardDataMap = new Map(wardStats.map(d => [d.ku_name, d]));

    // Filter to only the 23 special wards (区) — excludes western cities, towns, islands
    const wardFeatures = {
      type: 'FeatureCollection',
      features: tokyoGeoJSON.features.filter(f =>
        f.properties.ward_ja && f.properties.ward_ja.endsWith('区')
      ),
    };

    const svg = d3
      .select(svgRef.current)
      .append('svg')
      .attr('viewBox', `0 0 ${width} ${height}`)
      .style('width', '100%')
      .style('height', 'auto')
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const projection = d3.geoMercator()
      .fitExtent([[0, 0], [chartWidth, chartHeight - 70]], wardFeatures);

    const pathGenerator = d3.geoPath().projection(projection);

    // Color scale
    const colorScale = d3
      .scaleSequential()
      .domain([0, d3.max(wardStats, d => d.avgRent)])
      .interpolator(d3.interpolateYlOrRd);

    // Tooltip
    const tooltip = d3
      .select('body')
      .append('div')
      .attr('class', 'tokyo-map-tooltip')
      .style('position', 'absolute')
      .style('visibility', 'hidden')
      .style('background-color', 'rgba(0, 0, 0, 0.9)')
      .style('color', '#fff')
      .style('padding', '12px')
      .style('border-radius', '8px')
      .style('font-size', '13px')
      .style('pointer-events', 'none')
      .style('z-index', '1000')
      .style('box-shadow', '0 4px 6px rgba(0,0,0,0.3)');

    // Draw ward boundaries and fill with heatmap colors
    svg
      .selectAll('path')
      .data(wardFeatures.features)
      .enter()
      .append('path')
      .attr('d', pathGenerator)
      .attr('class', 'ward')
      .attr('fill', d => {
        // GeoJSON uses "中野区" format; CSV ku_name uses "中野" (no 区 suffix)
        const wardName = (d.properties.ward_ja ?? '').replace('区', '');
        const wardData = wardDataMap.get(wardName);
        return wardData ? colorScale(wardData.avgRent) : '#ccc';
      })
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        const wardName = (d.properties.ward_ja ?? '').replace('区', '');
        const wardData = wardDataMap.get(wardName);

        d3.select(this)
          .attr('stroke', '#000')
          .attr('stroke-width', 3)
          .raise();

        if (wardData) {
          tooltip
            .style('visibility', 'visible')
            .html(`
              <strong>${wardData.ku_name}</strong><br/>
              <span style="color: #ffd700;">Avg Rent: ¥${(wardData.avgRent * 10000).toLocaleString()}/month</span><br/>
              Min: ¥${(wardData.minRent * 10000).toLocaleString()}<br/>
              Max: ¥${(wardData.maxRent * 10000).toLocaleString()}<br/>
              <span style="color: #88ff88;">Listings: ${wardData.count}</span>
            `);
        } else {
          tooltip
            .style('visibility', 'visible')
            .html(`
              <strong>${d.properties.ward_ja ?? wardName}</strong><br/>
              <span style="color: #ff8888;">No data available</span>
            `);
        }
      })
      .on('mousemove', function(event) {
        tooltip
          .style('top', (event.pageY - 10) + 'px')
          .style('left', (event.pageX + 10) + 'px');
      })
      .on('mouseout', function() {
        d3.select(this)
          .attr('stroke', '#fff')
          .attr('stroke-width', 1.5);

        tooltip.style('visibility', 'hidden');
      });

    // Add title
    svg
      .append('text')
      .attr('x', chartWidth / 2)
      .attr('y', -20)
      .attr('text-anchor', 'middle')
      .style('font-size', '18px')
      .style('font-weight', 'bold')
      .text('Tokyo Ward Rent Heatmap');

    // Create legend
    const legendWidth = 300;
    const legendHeight = 15;
    const legendX = chartWidth - legendWidth - 20;
    const legendY = chartHeight - 40;

    const legendScale = d3.scaleLinear()
      .domain(colorScale.domain())
      .range([0, legendWidth]);

    const legendAxis = d3.axisBottom(legendScale)
      .ticks(5)
      .tickFormat(d => `¥${d.toFixed(1)}`);

    // Legend gradient
    const defs = svg.append('defs');
    const linearGradient = defs
      .append('linearGradient')
      .attr('id', 'legend-gradient');

    const numStops = 10;
    for (let i = 0; i <= numStops; i++) {
      linearGradient
        .append('stop')
        .attr('offset', `${(i / numStops) * 100}%`)
        .attr('stop-color', colorScale(colorScale.domain()[0] + (i / numStops) * (colorScale.domain()[1] - colorScale.domain()[0])));
    }

    const legendGroup = svg
      .append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${legendX},${legendY})`);

    legendGroup
      .append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', 'url(#legend-gradient)')
      .attr('stroke', '#333')
      .attr('stroke-width', 1);

    legendGroup
      .append('g')
      .attr('transform', `translate(0,${legendHeight})`)
      .call(legendAxis)
      .selectAll('text')
      .style('font-size', '11px');

    legendGroup
      .append('text')
      .attr('x', legendWidth / 2)
      .attr('y', -5)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('font-weight', 'bold')
      .text('Average Rent (万円/month)');

    // Cleanup tooltip on unmount
    return () => {
      tooltip.remove();
    };

  }, [data, width, height, tokyoGeoJSON]);

  if (loading) {
    return (
      <div className="tokyo-map-loading">
        <div className="loading-spinner"></div>
        <p>Loading Tokyo map...</p>
      </div>
    );
  }

  if (!tokyoGeoJSON) {
    return (
      <div className="tokyo-map-error">
        <p>Unable to load Tokyo map data. Please try again later.</p>
      </div>
    );
  }

  return <div ref={svgRef} className="tokyo-map-container"></div>;
};

export default TokyoMapHeatmap;
