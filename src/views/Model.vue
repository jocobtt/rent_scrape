<template>
<div>
  <div class="model">
    <div class="model__title">
      <h1>🏡 Calculate your rent! </h1>
    </div>
    <div class="model__content">
      <div class="model__content__left">
        <div class="model__content__left__image">
          <img :src="image" alt="model image" />
        </div>
      </div>
      <div class="model__content__right">
        <div class="model__content__right__input">
          <h2>Input apartment parameters here</h2>
          <label for="sqrM">Square Meters: </label>
            <input type="text" v-model="sqrM" class="form-control"/>
            <label for="ApartmentType">Apartment Type: </label>
            <input type="text" v-model="ApartmentType" class="form-control"/>
            <label for="KuName">Ku name: </label>
            <input type="text" v-model="KuName" class="form-control"/>
            <label for="Yearbuilt">Year built: </label>
            <input type="text" v-model="Yearbuilt" class="form-control"/>
            <label for="Floor">Floor: </label>
            <input type="text" v-model="Floor" class="form-control"/>
            <label for="Distance">Distance: </label>
            <input type="text" v-model="Distance" class="form-control"/>
            <label for="Time">Time: </label>
            <input type="text" v-model="Time" class="form-control"/>
            
        </div>
        <div class="model__content__right__price">
          <p>Model disclaimer/reminder goes here</p>
        </div>
        <div class="model__content__right__button">
          <button type ="calculate" class = "btn btn-success btn-block mt-3">Calculate Rent!</button>
        </div>
        <div v-if="rentPrice" class="rentPrice">
          <p>Your estimated rent is: {{rentPrice}}</p>
        </div>
      </div>
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
        KuName: "",
        rentPrice: "",
        Yearbuilt: "",
        Floor: "",
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
<!-- want to eventually return prediction confidence etc. -->
<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #ff0000;
  max-width: 720px;
  margin: 100px auto;
  padding: 0 20px;
}
svg {
  /* important for responsiveness */
  display: block;
  fill: none;
  stroke: none;
  width: 100%;
  height: 100%;
  overflow: visible;
  background: #fff;
}
.buttons {
  margin-top: 2rem;
}
</style>