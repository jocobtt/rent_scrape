<template>
    <div ref="barChart"></div>
  </template>
  
  <script>
  import { onMounted, ref } from 'vue';
  import * as d3 from 'd3';
  
  // set dimensions and margins of the graph
  var margin = { top: 30, right: 30, bottom: 70, left: 60 },
    width = 460 - margin.left - margin.right,
    height = 400 - margin.top - margin.bottom;
  
  export default {
    name: 'BarChart',
    props: ['data'],
    setup(props) {
      const barChart = ref(null);
  
      onMounted(async () => {
        // parse the data
        const data = await d3.csv(props.data, d3.autoType);
  
        // create the SVG element
        const svg = d3
          .select(barChart.value)
          .append('svg')
          .attr('width', width + margin.left + margin.right)
          .attr('height', height + margin.top + margin.bottom)
          .append('g')
          .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');
  
        // x axis
        const x = d3
          .scaleBand()
          .range([0, width])
          .domain(data.map((d) => d.Country))
          .padding(0.2);
        svg
          .append('g')
          .attr('transform', 'translate(0,' + height + ')')
          .call(d3.axisBottom(x))
          .selectAll('text')
          .attr('transform', 'translate(-10,0)rotate(-45)')
          .style('text-anchor', 'end');
  
        // add y axis
        const y = d3.scaleLinear().domain([0, 13000]).range([height, 0]);
        svg.append('g').call(d3.axisLeft(y));
  
        // bars
        svg
          .selectAll('mybar')
          .data(data)
          .enter()
          .append('rect')
          .attr('x', function (d) {
            return x(d.Country);
          })
          .attr('y', function (d) {
            return y(d.Value);
          })
          .attr('width', x.bandwidth())
          .attr('height', function (d) {
            return height - y(d.Value);
          })
          .attr('fill', '#69b3a2');
      });
  
      return { barChart };
    },
  };
  </script>
// https://d3-graph-gallery.com/graph/barplot_basic.html