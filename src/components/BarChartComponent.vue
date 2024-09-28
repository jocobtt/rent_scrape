<template>
  <div ref="barChart"></div>
</template>

<script>
import { onMounted, onUnmounted, ref, watch } from 'vue';
import * as d3 from 'd3';

export default {
  name: 'BarChart',
  props: ['csvData'],
  setup(props) {
    const barChart = ref(null);
    let svg = null;

    const drawChart = () => {
      if (svg) {
        svg.remove(); // Clear previous chart if it exists
      }

      const margin = { top: 30, right: 30, bottom: 70, left: 60 },
        width = 460 - margin.left - margin.right,
        height = 400 - margin.top - margin.bottom;

      // Append the SVG object to the div with the ref barChart
      svg = d3
        .select(barChart.value)
        .append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

      // Parse the data
      const data = props.csvData;

      // X axis
      const x = d3
        .scaleBand()
        .range([0, width])
        .domain(data.map((d) => d.ku_name))
        .padding(0.2);
      svg
        .append('g')
        .attr('transform', 'translate(0,' + height + ')')
        .call(d3.axisBottom(x))
        .selectAll('text')
        .attr('transform', 'translate(-10,0)rotate(-45)')
        .style('text-anchor', 'end');

      // Y axis
      const y = d3.scaleLinear().domain([0, d3.max(data, (d) => d.rent_price)]).range([height, 0]);
      svg.append('g').call(d3.axisLeft(y));

      // Bars
      svg
        .selectAll('mybar')
        .data(data)
        .enter()
        .append('rect')
        .attr('x', (d) => x(d.ku_name))
        .attr('y', (d) => y(d.rent_price))
        .attr('width', x.bandwidth())
        .attr('height', (d) => height - y(d.rent_price))
        .attr('fill', '#69b3a2');
    };

    onMounted(drawChart);

    watch(
      () => props.csvData,
      () => {
        drawChart(); // Redraw chart if the data changes
      }
    );

    onUnmounted(() => {
      if (svg) {
        svg.remove();
      }
    });

    return { barChart };
  },
};
</script>

 <!-- https://d3-graph-gallery.com/graph/barplot_basic.html -->