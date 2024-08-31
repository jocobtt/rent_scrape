<template>
    <div id="app">
        <h1>🗼Tokyo Apartment Data</h1>
        <div v-if="loading">Loading...</div>
        <BarChart v-else :csvData="dataset" />
    </div>
</template>

<script>
import BarChart from "../components/BarChartComponent.vue";
import axios from 'axios';

export default{
    name: "BarChart",
    components: {
        BarChart,
    },
    data() {
        return {
            loading: true,
            dataset: null,
        };
    },
    async created() {
        try {
            const response = await axios.get('http://localhost:3000/api/data'); // have this be our fastapi endpoint for getting the data
            this.dataset = response.data;
            this.loading = false;
        } catch (error) {
            console.error(error);
        }
    }
};
</script>