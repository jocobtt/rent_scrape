<template>
  <div ref="chart">
    <svg ref="svgRef">
      <g class="x-axis" />
      <g class="y-axis" />  
  </div>
</template>

<script>
import { onMounted, ref, watchEffect } from "vue";

import * as d3 from 'd3';
import useResizeObserver from "../use/resizeObserver";

export default {
  mounted() {
    const data = [10, 20, 30, 40, 50]

    const svg = d3.select(this.$refs.chart)
      .append('svg')
      .attr('width', 400)
      .attr('height', 300)

    svg.selectAll('rect')
      .data(data)
      .enter()
      .append('rect')
      .attr('x', (d, i) => i * 60)
      .attr('y', 50)
      .attr('width', 50)
      .attr('height', d => d)
      .attr('fill', 'blue')
  }
}
</script>
