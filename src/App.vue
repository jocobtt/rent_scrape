<template>
  <div class="container">
    <NavBar />
  </div>
  <div id="app">
    <routerView></routerView>
  </div>
  <div>  <Footer /> </div>

</template>

<script>
import Footer from "./components/FooterApp.vue";
import NavBar from "./components/NavBar.vue";
export default {
  name: "App",
  components: {
    Footer,
    NavBar,
  },
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
