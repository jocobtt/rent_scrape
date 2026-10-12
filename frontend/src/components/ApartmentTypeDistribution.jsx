import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import useResizeObserver from '../hooks/useResizeObserver';

const ApartmentTypeDistribution = ({ data }) => {
  const chartRef = useRef(null);
  const { resizeRef, dimensions } = useResizeObserver();
  const totalWidth = dimensions?.width > 0 ? dimensions.width : 600;

  useEffect(() => {
    if (!data || data.length === 0 || !chartRef.current) return;

    // Clear previous chart
    d3.select(chartRef.current).selectAll('*').remove();

    // Count apartments by type
    const typeCounts = d3.rollup(
      data,
      v => v.length,
      d => d.apartment_type
    );

    const chartData = Array.from(typeCounts, ([type, count]) => ({
      type,
      count
    })).sort((a, b) => b.count - a.count);

    const margin = { top: 20, right: 30, bottom: 80, left: 60 };
    const width = totalWidth - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const svg = d3
      .select(chartRef.current)
      .append('svg')
      .attr('width', totalWidth)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // X axis
    const x = d3
      .scaleBand()
      .range([0, width])
      .domain(chartData.map(d => d.type))
      .padding(0.3);

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
      .domain([0, d3.max(chartData, d => d.count)])
      .range([height, 0]);

    svg.append('g').call(d3.axisLeft(y));

    // Bars
    svg
      .selectAll('rect')
      .data(chartData)
      .enter()
      .append('rect')
      .attr('x', d => x(d.type))
      .attr('y', d => y(d.count))
      .attr('width', x.bandwidth())
      .attr('height', d => height - y(d.count))
      .attr('fill', '#48bb78')
      .on('mouseover', function() {
        d3.select(this).attr('fill', '#2f855a');
      })
      .on('mouseout', function() {
        d3.select(this).attr('fill', '#48bb78');
      });

    // Add value labels on top of bars
    svg
      .selectAll('text.label')
      .data(chartData)
      .enter()
      .append('text')
      .attr('class', 'label')
      .attr('x', d => x(d.type) + x.bandwidth() / 2)
      .attr('y', d => y(d.count) - 5)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .text(d => d.count);

  }, [data, totalWidth]);

  return (
    <div ref={resizeRef} style={{ width: '100%' }}>
      <div ref={chartRef}></div>
    </div>
  );
};

export default ApartmentTypeDistribution;
