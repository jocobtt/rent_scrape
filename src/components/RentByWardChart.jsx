import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

const RentByWardChart = ({ data }) => {
  const chartRef = useRef(null);

  useEffect(() => {
    if (!data || data.length === 0 || !chartRef.current) return;

    // Clear previous chart
    d3.select(chartRef.current).selectAll('*').remove();

    // Group data by ward and calculate average rent
    const wardData = d3.rollup(
      data,
      v => d3.mean(v, d => +d.rent_price),
      d => d.ku_name
    );

    const chartData = Array.from(wardData, ([ward, avgRent]) => ({
      ward,
      avgRent: avgRent || 0
    })).sort((a, b) => b.avgRent - a.avgRent);

    const margin = { top: 20, right: 30, bottom: 80, left: 60 };
    const width = 600 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const svg = d3
      .select(chartRef.current)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // X axis
    const x = d3
      .scaleBand()
      .range([0, width])
      .domain(chartData.map(d => d.ward))
      .padding(0.2);

    svg
      .append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x))
      .selectAll('text')
      .attr('transform', 'translate(-10,0)rotate(-45)')
      .style('text-anchor', 'end');

    // Y axis
    const y = d3
      .scaleLinear()
      .domain([0, d3.max(chartData, d => d.avgRent)])
      .range([height, 0]);

    svg.append('g').call(d3.axisLeft(y));

    // Bars
    svg
      .selectAll('rect')
      .data(chartData)
      .enter()
      .append('rect')
      .attr('x', d => x(d.ward))
      .attr('y', d => y(d.avgRent))
      .attr('width', x.bandwidth())
      .attr('height', d => height - y(d.avgRent))
      .attr('fill', '#4299e1')
      .on('mouseover', function() {
        d3.select(this).attr('fill', '#2b6cb0');
      })
      .on('mouseout', function() {
        d3.select(this).attr('fill', '#4299e1');
      });

  }, [data]);

  return <div ref={chartRef}></div>;
};

export default RentByWardChart;
