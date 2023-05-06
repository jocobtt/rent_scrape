<template>
    <div ref="scatterPlot"></div>
</template>

<script>
import {onMounted, onUnmounted, ref } from 'vue';
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
            const data = await d3.csvParse(props.csvData,d3.autoType);

            const svg = d3
                .select(scatterPlot.value)
                .append('svg')
                .attr('width', width + margin.left + margin.right)
                .attr('height', height + margin.top + margin.bottom)
                .append('g')
                .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

            const x = d3
                .scaleLinear()
                .domain([0, 100])
                .range([0, width]);
            svg
                .append('g')
                .attr('transform', 'translate(0,' + height + ')')
                .call(d3.axisBottom(x));

            const y = d3.scaleLinear().domain([0, 13000]).range([height, 0]);
            svg.append('g').call(d3.axisLeft(y));

            svg
                .selectAll('mybar')
                .data(data)
                .enter()
                .append('circle')
                .attr('cx', function (d) {
                    return x(d.age);
                })
                .attr('cy', function (d) {
                    return y(d.rent_price);
                })
                .attr('r', 8)
                .style('fill', '#69b3a2');

            // Add X axis label:
            svg
                .append('text')
                .attr('text-anchor', 'end')
                .attr('x', width / 2 + margin.left)
                .attr('y', height + margin.top + 20)
                .text('Age');

            // Y axis label:
            svg
                .append('text')
                .attr('text-anchor', 'end')
                .attr('transform', 'rotate(-90)')
                .attr('y', -margin.left + 20)
                .attr('x', -margin.top - height / 2 + 20)
                .text('Rent Price');

            // Title
            svg
                .append('text')
                .attr()
                .attr('text-anchor', 'start')
                .attr('x', width / 2 + margin.left)
                .attr('y', -margin.top / 2)
                .text('Rent Price vs Age');
        });// end of onMounted
        
        onUnmounted(() => {
            svg.remove();
        });

        return {
            scatterPlot
        };
    },
};
</script>