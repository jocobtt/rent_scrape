<template>
    <div ref="scatterPlot"></div>
  </template>
  
  <script>
  import { onMounted, onUnmounted, ref } from 'vue';
  import * as d3 from 'd3';
  
  var margin = { top: 30, right: 30, bottom: 70, left: 60 },
      width = 460 - margin.left - margin.right,
      height = 400 - margin.top - margin.bottom;
  
  export default {
    name: 'ScatterPlot',
    props: ['csvData'],
    setup(props) {
      const scatterPlot = ref(null);
      let svg = null;
  
      onMounted(async () => {
        // Parse the CSV data
        const data = await d3.csvParse(props.csvData, d3.autoType);
  
        svg = d3
          .select(scatterPlot.value)
          .append('svg')
          .attr('width', width + margin.left + margin.right)
          .attr('height', height + margin.top + margin.bottom)
          .append('g')
          .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');
  
        // X scale for square meters (sqr_m)
        const x = d3
          .scaleLinear()
          .domain([0, d3.max(data, d => d.sqr_m)]) // Set domain based on data
          .range([0, width]);
  
        // Y scale for rent price
        const y = d3
          .scaleLinear()
          .domain([0, d3.max(data, d => d.rent_price)]) // Set domain based on data
          .range([height, 0]);
  
        // Create color scale for ku_name
        const color = d3.scaleOrdinal()
          .domain([...new Set(data.map(d => d.ku_name))]) // Unique ku_name values
          .range(d3.schemeCategory10);
  
        // X axis
        svg
          .append('g')
          .attr('transform', 'translate(0,' + height + ')')
          .call(d3.axisBottom(x));
  
        // Y axis
        svg.append('g').call(d3.axisLeft(y));
  
        // Add scatterplot points
        svg
          .selectAll('circle')
          .data(data)
          .enter()
          .append('circle')
          .attr('cx', d => x(d.sqr_m))
          .attr('cy', d => y(d.rent_price))
          .attr('r', 8)
          .style('fill', d => color(d.ku_name)) // Color by ku_name
          .style('opacity', 0.7);
  
        // Add X axis label
        svg
          .append('text')
          .attr('text-anchor', 'end')
          .attr('x', width / 2 + margin.left)
          .attr('y', height + margin.top + 20)
          .text('Square Meters (sqr_m)');
  
        // Add Y axis label
        svg
          .append('text')
          .attr('text-anchor', 'end')
          .attr('transform', 'rotate(-90)')
          .attr('y', -margin.left + 20)
          .attr('x', -margin.top - height / 2 + 20)
          .text('Rent Price (¥10,000)');
  
        // Add title
        svg
          .append('text')
          .attr('text-anchor', 'start')
          .attr('x', width / 2 + margin.left)
          .attr('y', -margin.top / 2)
          .style('font-size', '16px')
          .style('font-weight', 'bold')
          .text('Rent Price vs Square Meters by Ku');
      });
  
      onUnmounted(() => {
        svg.remove();
      });
  
      return {
        scatterPlot
      };
    },
  };
  </script>