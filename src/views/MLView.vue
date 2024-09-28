<!--Show shap output or mlflow graphics maybe? -->
<!-- <template>
    <div>
      <h1>MLflow Metrics</h1>
      <div v-if="loading">Loading...</div>
      <div v-else>
        <div v-for="run in mlflowData" :key="run.run_id">
          <h2>{{ run.experiment_name }} - {{ run.run_id }}</h2>
          <h3>Metrics</h3>
          <ul>
            <li v-for="(value, key) in run.metrics" :key="key">{{ key }}: {{ value }}</li>
          </ul>
          <h3>Parameters</h3>
          <ul>
            <li v-for="(value, key) in run.params" :key="key">{{ key }}: {{ value }}</li>
          </ul>
        </div>
      </div>
    </div>
  </template>
  
  <script>
  import axios from "axios";
  
  export default {
    data() {
      return {
        mlflowData: [],
        loading: true,
      };
    },
    async mounted() {
      try {
        const response = await axios.get("/api/mlflow-data");
        this.mlflowData = response.data;
      } catch (error) {
        console.error("Error fetching MLflow data:", error.message);
      } finally {
        this.loading = false;
      }
    },
  };
  </script> -->
  <!-- https://www.databricks.com/blog/2020/02/18/how-to-display-model-metrics-in-dashboards-using-the-mlflow-search-api.html -->