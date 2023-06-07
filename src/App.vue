<template>
  <v-app full-height full-width>
    <div class="container">
    <NavBar />
  </div>
  <v-main>
    <v-container>
      <router-view></router-view>
    </v-container>
  </v-main>
  <div>
    <Footer />
  </div>
  </v-app>
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
.buttons {
  margin-top: 2rem;
}
</style>
