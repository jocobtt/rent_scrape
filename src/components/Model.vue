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
          <h2>Input rent price here</h2>
          <label for="field1">Rent Price: </label>
            <input type="text" v-model="field1" class="form-control"/>
            <label for="field2">Number of Bedrooms: </label>
            <input type="text" v-model="field2" class="form-control"/>
            <label for="field3">Number of Bathrooms: </label>
            <input type="text" v-model="field3" class="form-control"/>
        </div>
        <div class="model__content__right__price">
          <p>$3</p>
        </div>
        <div class="model__content__right__button">
          <button type ="calculate" class = "btn btn-success btn-block mt-3">Calculate Rent!</button>
        </div>
        <div class="model__content__right__description">
          <p>test description</p>
        </div>
      </div>
    </div>
  </div>
</div>
</template>

<script>
export default {
    name: "Model",
    data() {
    return {
        field1: "",
        field2: "",
        field3: "",
        image: require("../assets/model.png"),
    };
    }
};