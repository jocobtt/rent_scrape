<template>
  <div id="home-page">
    <h1>Predict your rent! 🗼</h1>
    <div id="main-part">
      <p> This app is for visualizing and predicting rent in the greater Tokyo area! </p>
      <!-- change this to instead be a directory of sorts and intro -->
    </div>
  </div>
</template>
  
<script>
export default {
  name: "HomeView",
};
</script>
  
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