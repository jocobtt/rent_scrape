import React, { useRef, useEffect } from 'react';
import {
  select,
  line,
  scaleLinear,
  min,
  max,
  curveBasis,
  axisBottom,
  axisLeft,
} from 'd3';
import useResizeObserver from '../hooks/useResizeObserver';

const ResponsiveLineChart = ({ data }) => {
  const svgRef = useRef(null);
  const { resizeRef, dimensions } = useResizeObserver();

  useEffect(() => {
    if (!svgRef.current || !dimensions.width) return;

    const svg = select(svgRef.current);
    const { width, height } = dimensions;

    // Scales: map index / data values to pixel values on x-axis / y-axis
    const xScale = scaleLinear()
      .domain([0, data.length - 1])
      .range([0, width]);

    const yScale = scaleLinear()
      .domain([min(data), max(data)])
      .range([height, 0]);

    // Line generator: D3 method to transform an array of values to data points ("d") for a path element
    const lineGen = line()
      .curve(curveBasis)
      .x((value, index) => xScale(index))
      .y((value) => yScale(value));

    // Render path element with D3's General Update Pattern
    svg
      .selectAll('.line')
      .data([data])
      .join('path')
      .attr('class', 'line')
      .attr('stroke', 'green')
      .attr('fill', 'none')
      .attr('d', lineGen);

    // Render axes with help of scales
    const xAxis = axisBottom(xScale);
    svg
      .select('.x-axis')
      .style('transform', `translateY(${height}px)`)
      .call(xAxis);

    const yAxis = axisLeft(yScale);
    svg.select('.y-axis').call(yAxis);
  }, [data, dimensions]);

  return (
    <div ref={resizeRef}>
      <svg ref={svgRef} style={{ width: '100%', height: '400px' }}>
        <g className="x-axis" />
        <g className="y-axis" />
      </svg>
    </div>
  );
};

export default ResponsiveLineChart;
