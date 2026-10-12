import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import useResizeObserver from '../hooks/useResizeObserver';

const Histogram = ({ data, field = 'rent_price', numBins = 20, width: propWidth = 600, height = 400 }) => {
  const svgRef = useRef(null);
  const { resizeRef, dimensions } = useResizeObserver();
  const width = dimensions?.width > 0 ? dimensions.width : propWidth;

  useEffect(() => {
    if (!data || data.length === 0 || !svgRef.current) return;

    // Clear previous chart
    d3.select(svgRef.current).selectAll('*').remove();

    const margin = { top: 20, right: 30, bottom: 60, left: 60 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    const svg = d3
      .select(svgRef.current)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Get values and create bins
    const values = data.map(d => d[field]).filter(v => !isNaN(v));

    const xScale = d3
      .scaleLinear()
      .domain([0, d3.max(values)])
      .range([0, chartWidth]);

    const histogram = d3.histogram()
      .domain(xScale.domain())
      .thresholds(xScale.ticks(numBins));

    const bins = histogram(values);

    const yScale = d3
      .scaleLinear()
      .domain([0, d3.max(bins, d => d.length)])
      .range([chartHeight, 0]);

    // Add bars
    svg
      .selectAll('rect')
      .data(bins)
      .enter()
      .append('rect')
      .attr('x', d => xScale(d.x0) + 1)
      .attr('y', d => yScale(d.length))
      .attr('width', d => Math.max(0, xScale(d.x1) - xScale(d.x0) - 2))
      .attr('height', d => chartHeight - yScale(d.length))
      .attr('fill', '#4A90E2')
      .attr('opacity', 0.7)
      .attr('stroke', '#fff')
      .attr('stroke-width', 1)
      .on('mouseover', function() {
        d3.select(this).attr('opacity', 1);
      })
      .on('mouseout', function() {
        d3.select(this).attr('opacity', 0.7);
      });

    // Add count labels on bars
    svg
      .selectAll('text.bar-label')
      .data(bins)
      .enter()
      .append('text')
      .attr('class', 'bar-label')
      .attr('x', d => xScale(d.x0) + (xScale(d.x1) - xScale(d.x0)) / 2)
      .attr('y', d => yScale(d.length) - 5)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('fill', '#333')
      .text(d => d.length > 0 ? d.length : '');

    // Add X axis
    const xAxis = svg
      .append('g')
      .attr('transform', `translate(0,${chartHeight})`)
      .call(d3.axisBottom(xScale).ticks(10));

    // Add X axis label
    xAxis
      .append('text')
      .attr('x', chartWidth / 2)
      .attr('y', 45)
      .attr('fill', 'currentColor')
      .attr('text-anchor', 'middle')
      .style('font-size', '14px')
      .text(getFieldLabel(field));

    // Add Y axis
    const yAxis = svg
      .append('g')
      .call(d3.axisLeft(yScale).ticks(8));

    // Add Y axis label
    yAxis
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -chartHeight / 2)
      .attr('y', -45)
      .attr('fill', 'currentColor')
      .attr('text-anchor', 'middle')
      .style('font-size', '14px')
      .text('Frequency');

    // Add gridlines
    svg
      .append('g')
      .attr('class', 'grid')
      .attr('opacity', 0.1)
      .call(
        d3
          .axisLeft(yScale)
          .tickSize(-chartWidth)
          .tickFormat('')
      );

  }, [data, field, numBins, width, height]);

  return (
    <div ref={resizeRef} style={{ width: '100%' }}>
      <div ref={svgRef}></div>
    </div>
  );
};

const getFieldLabel = (field) => {
  const labels = {
    rent_price: 'Rent Price (万円/month)',
    sqr_m: 'Size (m²)',
    eki_walk: 'Walk to Station (minutes)',
    age: 'Building Age (years)',
    floor: 'Floor',
    price_per_sqm: 'Price per m² (万円)',
  };
  return labels[field] || field;
};

export default Histogram;
