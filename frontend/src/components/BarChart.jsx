import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import './BarChart.css';

const BarChart = ({ csvData }) => {
  const barChartRef = useRef(null);

  useEffect(() => {
    if (!csvData || !barChartRef.current) return;

    // Clear previous chart
    d3.select(barChartRef.current).selectAll('*').remove();

    const margin = { top: 30, right: 30, bottom: 70, left: 60 };
    const width = 460 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    // Create tooltip div
    const tooltip = d3
      .select(barChartRef.current)
      .append('div')
      .attr('class', 'bar-tooltip')
      .style('opacity', 0);

    // Append the SVG object to the div
    const svg = d3
      .select(barChartRef.current)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Parse the data
    const data = csvData;

    // X axis
    const x = d3
      .scaleBand()
      .range([0, width])
      .domain(data.map((d) => d.ku_name))
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
      .domain([0, d3.max(data, (d) => d.rent_price)])
      .range([height, 0]);

    svg.append('g').call(d3.axisLeft(y));

    // Bars with hover interactions
    svg
      .selectAll('mybar')
      .data(data)
      .enter()
      .append('rect')
      .attr('x', (d) => x(d.ku_name))
      .attr('y', (d) => y(d.rent_price))
      .attr('width', x.bandwidth())
      .attr('height', (d) => height - y(d.rent_price))
      .attr('fill', '#69b3a2')
      .attr('class', 'bar')
      .on('mouseover', function (event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('fill', '#4a9380');

        tooltip.transition().duration(200).style('opacity', 0.9);

        const rentValue = parseFloat(d.rent_price);
        const rentYen = (rentValue * 10000).toLocaleString();

        tooltip
          .html(`
            <strong>${d.ku_name}</strong><br/>
            Rent: ¥${rentYen}<br/>
            (${rentValue}万円/month)
          `)
          .style('left', event.pageX + 10 + 'px')
          .style('top', event.pageY - 28 + 'px');
      })
      .on('mouseout', function () {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('fill', '#69b3a2');

        tooltip.transition().duration(500).style('opacity', 0);
      });
  }, [csvData]);

  return <div ref={barChartRef}></div>;
};

export default BarChart;
