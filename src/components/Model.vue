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
    };
  },
    methods: {
      async callModel() {
        try {
          const response = await axios.get('http://localhost:5000/predict', {
            params: {
              sqrM: this.sqrM,
              ApartmentType: this.ApartmentType,
              KuName: this.KuName,
            }
          });
          this.rentPrice = response.data;
        } catch (error) {
          console.log(error);
        }
    }
  },
};