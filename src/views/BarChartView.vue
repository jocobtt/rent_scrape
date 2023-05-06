<template>
    <div id="app">
      <h1>🗼Tokyo Apartment Data</h1>
        <BarChart :csvData="localCsvData" />
    </div>
</template>

<script>
import BarChart from "../components/BarChartComponent.vue";
import localCsvData from "../tokyo_model.csv";

export default{
    name: "BarChart",
    components: {
        BarChart,
    },
    data() {
        return {
            localCsvData,
        };
    },
};
</script>