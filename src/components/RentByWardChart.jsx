import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { getAverageRentByWard } from '../lib/dataUtils';

const RentByWardChart = ({ data }) => {
  const chartRef = useRef(null);
  const [selectedWard, setSelectedWard] = useState(null);

  useEffect(() => {
    if (!data || data.length === 0 || !chartRef.current) return;

    d3.select(chartRef.current).selectAll('*').remove();

    const chartData = getAverageRentByWard(data);

    const margin = { top: 20, right: 70, bottom: 80, left: 60 };
    const width = 600 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const svg = d3
      .select(chartRef.current)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const x = d3
      .scaleBand()
      .range([0, width])
      .domain(chartData.map(d => d.ku_name))
      .padding(0.2);

    const y = d3
      .scaleLinear()
      .domain([0, d3.max(chartData, d => d.avgRent) * 1.15])
      .range([height, 0]);

    svg
      .append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x))
      .selectAll('text')
      .attr('transform', 'translate(-10,0)rotate(-45)')
      .style('text-anchor', 'end');

    svg.append('g').call(d3.axisLeft(y).tickFormat(d => `${d}万`));

    // Bars
    svg
      .selectAll('rect')
      .data(chartData)
      .enter()
      .append('rect')
      .attr('x', d => x(d.ku_name))
      .attr('y', d => y(d.avgRent))
      .attr('width', x.bandwidth())
      .attr('height', d => height - y(d.avgRent))
      .attr('fill', '#4299e1')
      .style('cursor', 'pointer')
      .on('mouseover', function() {
        d3.select(this).attr('fill', '#2b6cb0');
      })
      .on('mouseout', function() {
        d3.select(this).attr('fill', '#4299e1');
      })
      .on('click', function(event, d) {
        setSelectedWard(prev => prev?.ku_name === d.ku_name ? null : d);
      });

    // Value labels above bars
    svg
      .selectAll('.bar-label')
      .data(chartData)
      .enter()
      .append('text')
      .attr('class', 'bar-label')
      .attr('x', d => x(d.ku_name) + x.bandwidth() / 2)
      .attr('y', d => y(d.avgRent) - 4)
      .attr('text-anchor', 'middle')
      .style('font-size', '10px')
      .style('fill', '#555')
      .style('pointer-events', 'none')
      .text(d => `${d.avgRent.toFixed(1)}`);

  }, [data]);

  return (
    <div style={{ position: 'relative' }}>
      <div ref={chartRef}></div>

      {selectedWard && (
        <div style={{
          position: 'absolute',
          top: 10,
          right: 10,
          background: 'rgba(255,255,255,0.97)',
          border: '1px solid #e2e8f0',
          borderRadius: '8px',
          padding: '12px 16px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
          minWidth: '180px',
          fontSize: '13px',
        }}>
          <button
            onClick={() => setSelectedWard(null)}
            style={{ float: 'right', background: 'none', border: 'none', cursor: 'pointer', fontSize: '14px', color: '#888' }}
          >✕</button>
          <strong style={{ display: 'block', marginBottom: '8px' }}>{selectedWard.ku_name}</strong>
          <div>Avg Rent: <strong>¥{(selectedWard.avgRent * 10000).toLocaleString()}/mo</strong></div>
          <div>Min: ¥{(selectedWard.minRent * 10000).toLocaleString()}</div>
          <div>Max: ¥{(selectedWard.maxRent * 10000).toLocaleString()}</div>
          <div style={{ marginTop: '4px', color: '#666' }}>{selectedWard.count} listings</div>
        </div>
      )}
    </div>
  );
};

export default RentByWardChart;
