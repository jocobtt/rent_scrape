<template>
  <div ref="barChart"></div>
</template>

<script>
import { onMounted, onUnmounted, ref } from 'vue';
import * as d3 from 'd3';

var margin = { top: 30, right: 30, bottom: 70, left: 60 },
  width = 460 - margin.left - margin.right,
  height = 400 - margin.top - margin.bottom;

export default {
  name: 'BarChart',
  props: ['csvData'],
  setup(props) {
    const barChart = ref(null);
    let svg = null;

    onMounted(async () => {
      // Parse the CSV data
      const data = await d3.csvParse(props.csvData,d3.autoType);

      const svg = d3
        .select(barChart.value)
        .append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

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

      const y = d3.scaleLinear().domain([0, 13000]).range([height, 0]);
      svg.append('g').call(d3.axisLeft(y));

      svg
        .selectAll('mybar')
        .data(data)
        .enter()
        .append('rect')
        .attr('x', function (d) {
          return x(d.ku_name);
        })
        .attr('y', function (d) {
          return y(d.rent_price);
        })
        .attr('width', x.bandwidth())
        .attr('height', function (d) {
          return height - y(d.rent_price);
        })
        .attr('fill', '#69b3a2');
    });

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