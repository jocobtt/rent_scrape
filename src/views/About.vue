<template>
    <div class="flex flex-row items-center">
      <h1 class="text-2xl font-bold">About this project</h1>
    </div>
    <div class="text-gray-600">
      <p>
        This project is a vue/vite based application with a fastapi backend hosted on Vercel.
        From Suumo, a Japanese apartment listing site. We build our models using a few different methods, we have 
        one that is a regression based model and another that is a boosting based regression model.
      </p>
      <br>
      <p>
        We are generating the two models to help us predict a persons rent based on the apartment they are looking at. And then we also visualizing some parts of the data itself. We are using Dd3 to visualize the data.
      </p>
    </div>
  </template>
  
  <script>
  
  export default {
    name: "AboutView",
  };
  </script>