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
          <input type="number" v-model="sqrM" class="form-control" placeholder="Enter square meters" />
        </div>
        <div class="form-group">
          <label for="ApartmentType">Apartment Type:</label>
          <select v-model="ApartmentType" class="form-control">
            <option disabled value="">Please select one</option>
            <option v-for="type in apartmentTypes" :key="type">{{ type }}</option>
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
            <option v-for="type in Floors" :key="type">{{ type }}</option>
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
// your existing script code
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