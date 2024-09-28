<template>
    <div id="app" class="min-h-screen bg-gray-50 pt-20 flex flex-col items-center justify-center">
      <h1 class="text-4xl font-bold text-gray-900 mb-6">🗼 Tokyo Apartment Data</h1>
      <p class="text-lg text-gray-700 mb-4">Explore rent prices across various districts in Tokyo</p>
      
      <div v-if="loading" class="text-lg text-gray-600 animate-pulse">Loading data...</div>
      
      <BarChart v-else :csvData="dataset" class="w-full max-w-4xl mx-auto p-4 bg-white rounded-lg shadow-lg" />
      
      <div class="flex justify-center space-x-4 mt-8">
        <button @click="visualizeData" class="px-5 py-2 bg-blue-500 text-white font-semibold rounded-lg shadow-md hover:bg-blue-600 transition ease-in-out">
          Visualize Data
        </button>
        <button @click="filterData" class="px-5 py-2 bg-gray-500 text-white font-semibold rounded-lg shadow-md hover:bg-gray-600 transition ease-in-out">
          Filter Data
        </button>
      </div>
    </div>
  </template>
  
  <script>
  import BarChart from "../components/ScatterPlot.vue";
  import * as d3 from 'd3';
  
  export default {
    name: "ScatterPlotView",
    components: {
      ScatterPlot,
    },
    data() {
      return {
        loading: true,
        dataset: null,
      };
    },
    methods: {
      visualizeData() {
        // Handle data visualization logic
      },
      filterData() {
        // Handle data filtering logic
      }
    },
    async created() {
        try {
            // Load CSV file using d3.csv
            const data = await d3.csv('/tokyo_model.csv');
            this.dataset = data; // Store data for BarChart
            this.loading = false;
        } catch (error) {
            console.error("Error loading CSV file:", error);
        }
    }
  };
  </script>
  
  <style scoped>
  body {
    font-family: 'IBM Plex Mono', sans-serif;
    margin: 0;
    background-color: #f9fafb;
  }
  
  .animate-pulse {
    animation: pulse 1.5s infinite;
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  </style>