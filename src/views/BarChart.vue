<template>
    <div id="app">
      <h1>🗼Tokyo Apartment Data</h1>
        <BarChart :csvData="localCsvData" />
    </div>
</template>

<script>
import Barchart from "../components/Barchart.vue";
import localCsvData from "../tokyo_model.csv";

export default{
    name: "BarChart",
    components: {
        Barchart,
    },
    data() {
        return {
            localCsvData,
        };
    },
};
</script>