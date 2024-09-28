<template>
  <div class="model">
    <div class="model__title">
      <h1>🏡 Calculate your rent!</h1>
      <h2>Input apartment parameters here:</h2>
    </div>
    <div class="model__content">
      <div class="model__form">
        <div class="form-group">
          <label for="sqrM">Square Meters:</label>
          <input type="number" v-model="sqrM" class="form-control" placeholder="e.g., 80" min="0" max="1000" />
        </div>
        <div class="form-group">
          <label for="ApartmentType">Apartment Type:</label>
          <select v-model="ApartmentType" class="form-control">
            <option disabled value="">Please select one</option>
            <option v-for="type in apartmentTypes" :key="type" :value="type">
              {{ type }}
            </option>
          </select>
        </div>
        <div class="form-group">
          <label for="KuName">Ku Name:</label>
          <select v-model="KuName" class="form-control">
            <option disabled value="">Please select one</option>
            <option v-for="type in KuNames" :key="type">{{ type }}</option>

          </select>
        </div>
        <div class="form-group">
          <label for="Yearbuilt">Year Built:</label>
          <select v-model="Yearbuilt" class="form-control">
            <option disabled value="">Please select one</option>
            <option v-for="type in YearBuilts" :key="type">{{ type }}</option>
          </select>
        </div>
        <div class="form-group">
          <label for="Floor">Floor:</label>
          <select v-model="Floor" class="form-control">
            <option disabled value="">Please select one</option>
            <option v-for="floor in floorOptions" :key="floor" :value="floor">
              {{ floor === 0 ? 'Ground Floor' : floor }}
            </option>
          </select>
        </div>
        <button type="button" @click="callModel" class="btn btn-success">Calculate Rent!</button>
        <div v-if="rentPrice" class="rentPrice">
          <p>Your estimated rent is: {{rentPrice}}</p>
        </div>
        <p class="disclaimer">Model disclaimer/reminder goes here</p>
      </div>
    </div>
  </div>
</template>

<script>
// make axios api call 
import axios from 'axios';
export default {
    name: "Model",
    data() {
    return {
        sqrM: "",
        ApartmentType: "",
        apartmentTypes: ["1DK", "2K", "1K", "2DK", "1LDK", "2LDK", "Other"],
        KuName: "",
        KuNames: ["中野", "目黒", "文京", "新宿", "荒川", "墨田", "中央", "台東", "江東", "品川", "杉並", "港", "大田", "その他"],
        rentPrice: "",
        Yearbuilt: "",
        YearBuilts: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25"],
        Floor: "",
        floorOptions: Array.from({ length: 21 }, (_, i) => i),
        Distance: "",
        Time: ""
    };
  },
    methods: {
      async callModel() {
        try {
          const response = await axios.get('/predict', {
            params: {
              sqrM: this.sqrM,
              ApartmentType: this.ApartmentType,
              KuName: this.KuName,
              Yearbuilt: this.Yearbuilt,
              Floor: this.Floor,
              Distance: this.Distance,
              Time: this.Time
            }
          });
          this.rentPrice = response.data;
        } catch (error) {
          console.log(error);
        }
    }
  },
};
</script>

<style>
/* General layout adjustments */
.model {
  max-width: 600px;
  margin: 0 auto;
  padding: 20px;
  background-color: #fff;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
  border-radius: 8px;
}

.model__title {
  text-align: center;
  margin-bottom: 20px;
}

.model__content {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.model__form {
  width: 100%;
}

.form-group {
  margin-bottom: 15px;
}

.form-control {
  width: 100%;
  padding: 10px;
  margin-top: 5px;
  border: 1px solid #ccc;
  border-radius: 4px;
}

.btn-success {
  width: 100%;
  padding: 10px;
  background-color: #28a745;
  color: #fff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
  transition: background-color 0.3s;
}

.btn-success:hover {
  background-color: #218838;
}

.rentPrice {
  margin-top: 20px;
  font-weight: bold;
}

.disclaimer {
  margin-top: 10px;
  font-size: 0.9em;
  color: #666;
  text-align: center;
  font-style: italic;
}
</style>