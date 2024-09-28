<template>
  <div class="mlflow-view">
    <h1>MLflow Artifacts Visualization</h1>
    <div class="artifact-selector">
      <label for="artifact-select">Select an artifact:</label>
      <select id="artifact-select" v-model="selectedArtifact" @change="loadArtifact">
        <option v-for="artifact in artifacts" :key="artifact.path" :value="artifact.path">
          {{ artifact.name }}
        </option>
      </select>
    </div>
    <div class="artifact-visualization" v-if="artifactData">
      <component :is="visualizationComponent" :data="artifactData"></component>
    </div>
    <div v-else class="no-artifact-message">
      Please select an artifact to visualize.
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import LineChart from '@/components/visualizations/LineChart.vue';
import ScatterPlot from '@/components/visualizations/ScatterPlot.vue';
import DataTable from '@/components/visualizations/DataTable.vue';
import { Storage } from '@google-cloud/storage';

export default {
  name: 'MlflowView',
  components: {
    LineChart,
    ScatterPlot,
    DataTable,
  },
  setup() {
    const artifacts = ref([]);
    const selectedArtifact = ref(null);
    const artifactData = ref(null);

    const visualizationComponent = computed(() => {
      if (!artifactData.value) return null;
      switch (artifactData.value.type) {
        case 'line_chart':
          return 'LineChart';
        case 'scatter_plot':
          return 'ScatterPlot';
        default:
          return 'DataTable';
      }
    });

    const storage = new Storage();
    const bucketName = 'your-mlflow-artifacts-bucket'; 
    // change this to mlflow bucket

    const fetchArtifacts = async () => {
      try {
        const [files] = await storage.bucket(bucketName).getFiles();
        artifacts.value = files.map(file => ({
          name: file.name,
          path: file.name,
        }));
      } catch (error) {
        console.error('Error fetching artifacts from GCS:', error);
      }
    };

    const loadArtifact = async () => {
      if (!selectedArtifact.value) return;
      try {
        const file = storage.bucket(bucketName).file(selectedArtifact.value);
        const [content] = await file.download();
        
        // Parse the content based on file type
        const fileExtension = selectedArtifact.value.split('.').pop().toLowerCase();
        switch (fileExtension) {
          case 'json':
            artifactData.value = JSON.parse(content.toString());
            break;
          case 'csv':
            // You might want to use a CSV parsing library here
            artifactData.value = { type: 'table', data: content.toString() };
            break;
          default:
            artifactData.value = { type: 'text', data: content.toString() };
        }
      } catch (error) {
        console.error('Error loading artifact from GCS:', error);
      }
    };

    fetchArtifacts();

    return {
      artifacts,
      selectedArtifact,
      artifactData,
      visualizationComponent,
      loadArtifact,
    };
  },
};
</script>

<style scoped>
.mlflow-view {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
}

.artifact-selector {
  margin-bottom: 2rem;
}

.artifact-visualization {
  margin-top: 2rem;
}

.no-artifact-message {
  text-align: center;
  font-style: italic;
  color: #666;
}
</style>

