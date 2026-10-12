import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

const ScatterPlot = ({ csvData }) => {
  const scatterPlotRef = useRef(null);

  useEffect(() => {
    if (!csvData || !scatterPlotRef.current) return;

    // Clear previous chart
    d3.select(scatterPlotRef.current).selectAll('*').remove();

    const margin = { top: 30, right: 30, bottom: 70, left: 60 };
    const width = 460 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    // Parse the CSV data
    const data = d3.csvParse(csvData, d3.autoType);

    // Append the SVG object
    const svg = d3
      .select(scatterPlotRef.current)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // X axis
    const x = d3.scaleLinear().domain([0, 100]).range([0, width]);
    svg
      .append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x));

    // Y axis
    const y = d3.scaleLinear().domain([0, 13000]).range([height, 0]);
    svg.append('g').call(d3.axisLeft(y));

    // Scatter points
    svg
      .selectAll('circle')
      .data(data)
      .enter()
      .append('circle')
      .attr('cx', (d) => x(d.age))
      .attr('cy', (d) => y(d.rent_price))
      .attr('r', 8)
      .style('fill', '#69b3a2');

    // Add X axis label
    svg
      .append('text')
      .attr('text-anchor', 'end')
      .attr('x', width / 2 + margin.left)
      .attr('y', height + margin.top + 20)
      .text('Age');

    // Y axis label
    svg
      .append('text')
      .attr('text-anchor', 'end')
      .attr('transform', 'rotate(-90)')
      .attr('y', -margin.left + 20)
      .attr('x', -margin.top - height / 2 + 20)
      .text('Rent Price');

    // Title
    svg
      .append('text')
      .attr('text-anchor', 'start')
      .attr('x', width / 2 + margin.left)
      .attr('y', -margin.top / 2)
      .text('Rent Price vs Age');
  }, [csvData]);

  return <div ref={scatterPlotRef}></div>;
};

export default ScatterPlot;
