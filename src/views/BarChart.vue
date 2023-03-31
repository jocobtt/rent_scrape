<template>
    <div id="app">
      <h1>🗼Tokyo Apartment Data</h1>
        <BarChart csvUrl="path/to/our/csvfile.csv" />
    </div>
</template>

<script>
import Barchart from "../components/Barchart.vue";

export default {
    name: "BarChart",
    components: {
        Barchart,
    },
};
</script>