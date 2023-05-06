<template>
    <div ref="Histogram"></div>
</template>

<script>
import { onMounted, onUnmounted, ref } from 'vue';
import * as d3 from 'd3';

var margin = { top: 30, right: 30, bottom: 70, left: 60 },
  width = 460 - margin.left - margin.right,
  height = 400 - margin.top - margin.bottom;

export default {
    name: 'Histogram',
    props: ['csvData'],
    setup(props) {
        const Histogram = ref(null);
        let svg = null;

    onMounted(async () => {
        // Parse the CSV data
        const data = await d3.csvParse(props.csvData,d3.autoType);

        const svg = d3
            .select(Histogram.value)
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
        
        onUnmounted(() => {
            svg.remove();
        });
    });
    return {
        Histogram,
    };
    },
};
</script>