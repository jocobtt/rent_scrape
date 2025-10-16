import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import './InteractiveScatterPlot.css';

const InteractiveScatterPlot = ({ data }) => {
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);
  const [selectedPoint, setSelectedPoint] = useState(null);

  useEffect(() => {
    if (!data || data.length === 0 || !svgRef.current) return;

    // Clear previous chart
    d3.select(svgRef.current).selectAll('*').remove();

    const margin = { top: 20, right: 20, bottom: 60, left: 60 };
    const width = svgRef.current.clientWidth - margin.left - margin.right;
    const height = 500 - margin.top - margin.bottom;

    const svg = d3
      .select(svgRef.current)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3
      .scaleLinear()
      .domain([0, d3.max(data, d => d.sqr_m) * 1.1])
      .range([0, width]);

    const yScale = d3
      .scaleLinear()
      .domain([0, d3.max(data, d => d.rent_price) * 1.1])
      .range([height, 0]);

    const colorScale = d3
      .scaleOrdinal()
      .domain([...new Set(data.map(d => d.apartment_type))])
      .range(d3.schemeSet2);

    // Add gridlines
    svg
      .append('g')
      .attr('class', 'grid')
      .attr('opacity', 0.1)
      .call(
        d3
          .axisLeft(yScale)
          .tickSize(-width)
          .tickFormat('')
      );

    svg
      .append('g')
      .attr('class', 'grid')
      .attr('opacity', 0.1)
      .attr('transform', `translate(0,${height})`)
      .call(
        d3
          .axisBottom(xScale)
          .tickSize(-height)
          .tickFormat('')
      );

    // Axes
    svg
      .append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale))
      .append('text')
      .attr('x', width / 2)
      .attr('y', 40)
      .attr('fill', 'currentColor')
      .attr('text-anchor', 'middle')
      .style('font-size', '14px')
      .text('Square Meters (m²)');

    svg
      .append('g')
      .call(d3.axisLeft(yScale))
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', -45)
      .attr('fill', 'currentColor')
      .attr('text-anchor', 'middle')
      .style('font-size', '14px')
      .text('Rent (万円/month)');

    // Tooltip
    const tooltip = d3.select(tooltipRef.current);

    // Points
    svg
      .selectAll('circle')
      .data(data)
      .enter()
      .append('circle')
      .attr('cx', d => xScale(d.sqr_m))
      .attr('cy', d => yScale(d.rent_price))
      .attr('r', 6)
      .attr('fill', d => colorScale(d.apartment_type))
      .attr('opacity', 0.7)
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .style('cursor', 'pointer')
      .on('mouseover', function (event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', 10)
          .attr('opacity', 1);

        tooltip
          .style('opacity', 1)
          .style('left', event.pageX + 10 + 'px')
          .style('top', event.pageY - 10 + 'px')
          .html(`
            <div class="tooltip-content">
              <strong>${d.ku_name} - ${d.apartment_type}</strong>
              <div class="tooltip-row">
                <span>Rent:</span>
                <span>¥${(d.rent_price * 10000).toLocaleString()}/month</span>
              </div>
              <div class="tooltip-row">
                <span>Size:</span>
                <span>${d.sqr_m} m²</span>
              </div>
              <div class="tooltip-row">
                <span>Price/m²:</span>
                <span>¥${((d.rent_price * 10000) / d.sqr_m).toFixed(0)}</span>
              </div>
              <div class="tooltip-row">
                <span>Floor:</span>
                <span>${d.floor === 0 ? 'Ground' : d.floor}</span>
              </div>
              <div class="tooltip-row">
                <span>Walk to station:</span>
                <span>${d.eki_walk} min</span>
              </div>
            </div>
          `);
      })
      .on('mousemove', function (event) {
        tooltip
          .style('left', event.pageX + 10 + 'px')
          .style('top', event.pageY - 10 + 'px');
      })
      .on('mouseout', function () {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', 6)
          .attr('opacity', 0.7);

        tooltip.style('opacity', 0);
      })
      .on('click', function (event, d) {
        setSelectedPoint(d);
      });

    // Legend
    const legend = svg
      .append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${width - 150}, 10)`);

    const apartmentTypes = [...new Set(data.map(d => d.apartment_type))];

    apartmentTypes.slice(0, 6).forEach((type, i) => {
      const legendRow = legend
        .append('g')
        .attr('transform', `translate(0, ${i * 20})`);

      legendRow
        .append('circle')
        .attr('r', 5)
        .attr('fill', colorScale(type))
        .attr('opacity', 0.7);

      legendRow
        .append('text')
        .attr('x', 12)
        .attr('y', 5)
        .style('font-size', '12px')
        .text(type);
    });
  }, [data]);

  return (
    <div className="interactive-scatter-plot">
      <div ref={svgRef} className="scatter-plot-container"></div>
      <div ref={tooltipRef} className="scatter-tooltip"></div>

      {selectedPoint && (
        <div className="selected-info">
          <button className="close-btn" onClick={() => setSelectedPoint(null)}>
            ✕
          </button>
          <h3>{selectedPoint.ku_name} - {selectedPoint.apartment_type}</h3>
          <p><strong>Address:</strong> {selectedPoint.address}</p>
          <p><strong>Rent:</strong> ¥{(selectedPoint.rent_price * 10000).toLocaleString()}/month</p>
          <p><strong>Size:</strong> {selectedPoint.sqr_m} m²</p>
          <p><strong>Floor:</strong> {selectedPoint.floor === 0 ? 'Ground' : selectedPoint.floor}</p>
          <p><strong>Building Age:</strong> {selectedPoint.age} years</p>
          <p><strong>Walk to Station:</strong> {selectedPoint.eki_walk} minutes</p>
        </div>
      )}
    </div>
  );
};

export default InteractiveScatterPlot;
