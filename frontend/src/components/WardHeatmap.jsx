import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { getAverageRentByWard } from '../lib/dataUtils';
import useResizeObserver from '../hooks/useResizeObserver';

// One line of text per ward, so the rows are sized for it rather than squeezed into a fixed height
const ROW_HEIGHT = 26;

// Narrower than this the ward labels and the bar no longer both fit
const MIN_WIDTH = 280;

const WardHeatmap = ({ data }) => {
  const svgRef = useRef(null);
  // Drawn at the width its container has, redrawn when that changes (a fixed 700 px made phones scroll sideways)
  const { resizeRef, dimensions } = useResizeObserver();
  const width = dimensions.width ? Math.max(MIN_WIDTH, Math.floor(dimensions.width)) : 0;

  useEffect(() => {
    if (!width || !data || data.length === 0 || !svgRef.current) return;

    // Clear previous chart
    d3.select(svgRef.current).selectAll('*').remove();

    const margin = { top: 30, right: 20, bottom: 20, left: 70 };
    const chartWidth = width - margin.left - margin.right;

    // Get ward statistics
    const wardStats = getAverageRentByWard(data);

    const chartHeight = wardStats.length * ROW_HEIGHT;
    const height = chartHeight + margin.top + margin.bottom;

    const svg = d3
      .select(svgRef.current)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const yScale = d3
      .scaleBand()
      .domain(wardStats.map(d => d.ku_name))
      .range([0, chartHeight])
      .padding(0.1);

    const cellWidth = chartWidth;

    const colorScale = d3
      .scaleSequential()
      .domain([0, d3.max(wardStats, d => d.avgRent)])
      .interpolator(d3.interpolateYlOrRd);

    // Tooltip — remove any stale instance before creating a new one
    d3.selectAll('.heatmap-tooltip').remove();
    const tooltip = d3
      .select('body')
      .append('div')
      .attr('class', 'heatmap-tooltip')
      .style('position', 'absolute')
      .style('visibility', 'hidden')
      .style('background-color', 'rgba(0, 0, 0, 0.8)')
      .style('color', '#fff')
      .style('padding', '10px')
      .style('border-radius', '5px')
      .style('font-size', '13px')
      .style('pointer-events', 'none')
      .style('z-index', '1000');

    // Draw cells
    svg
      .selectAll('rect')
      .data(wardStats)
      .enter()
      .append('rect')
      .attr('y', d => yScale(d.ku_name))
      .attr('x', 0)
      .attr('width', cellWidth)
      .attr('height', yScale.bandwidth())
      .attr('fill', d => colorScale(d.avgRent))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        d3.select(this)
          .attr('stroke', '#000')
          .attr('stroke-width', 3);

        tooltip
          .style('visibility', 'visible')
          .html(`
            <strong>${d.ku_name}</strong><br/>
            Avg Rent: ¥${(d.avgRent * 10000).toLocaleString()}/month<br/>
            Min: ¥${(d.minRent * 10000).toLocaleString()}<br/>
            Max: ¥${(d.maxRent * 10000).toLocaleString()}<br/>
            Listings: ${d.count}
          `);
      })
      .on('mousemove', function(event) {
        tooltip
          .style('top', (event.pageY - 10) + 'px')
          .style('left', (event.pageX + 10) + 'px');
      })
      .on('mouseout', function() {
        d3.select(this)
          .attr('stroke', '#fff')
          .attr('stroke-width', 2);

        tooltip.style('visibility', 'hidden');
      });

    // One label per ward: average rent, then how many listings it is based on
    const maxRent = d3.max(wardStats, w => w.avgRent);
    const onDark = d => d.avgRent > maxRent * 0.6;

    const labels = svg
      .selectAll('text.value-label')
      .data(wardStats)
      .enter()
      .append('text')
      .attr('class', 'value-label')
      .attr('x', cellWidth / 2)
      .attr('y', d => yScale(d.ku_name) + yScale.bandwidth() / 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .style('font-size', '12px')
      .style('fill', d => (onDark(d) ? '#fff' : '#000'))
      .style('pointer-events', 'none');

    labels.append('tspan')
      .style('font-weight', 'bold')
      .text(d => `${d.avgRent.toFixed(1)}万円`);
    labels.append('tspan')
      .attr('dx', 8)
      .style('font-size', '11px')
      .style('fill', d => (onDark(d) ? 'rgba(255,255,255,0.85)' : '#555'))
      .text(d => `${d.count.toLocaleString()} listings`);

    // Add Y axis (ward names)
    svg
      .append('g')
      .call(d3.axisLeft(yScale))
      .selectAll('text')
      .style('font-size', '12px');

    // Add title
    svg
      .append('text')
      .attr('x', cellWidth / 2)
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .style('font-size', '16px')
      .style('font-weight', 'bold')
      .text('Average Rent by Ward (万円/month)');

    // Cleanup tooltip on unmount
    return () => {
      tooltip.remove();
    };

  }, [data, width]);

  return (
    <div ref={resizeRef} style={{ overflow: 'auto' }}>
      <div ref={svgRef} />
    </div>
  );
};

export default WardHeatmap;
