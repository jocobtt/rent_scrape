<template>
  <v-container id="home-page" fluid>
    <v-row justify="center" class="hero-section">
      <v-col cols="12" sm="10" md="8" class="text-center">
        <h1 class="title">Predict your rent! 🗼</h1>
        <p class="intro-text">
          This app is for visualizing and predicting rent in the greater Tokyo area!
        </p>
        <router-link to="/model">
          <v-btn class="cta-button" large>Get Started</v-btn>
        </router-link>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
export default {
  name: "HomeView",
};
</script>

<style scoped>
/* Base styles for reset */
body, html {
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  font-family: 'Roboto', sans-serif;
  background-color: #f4f7f6;
}

/* Hero section styling */
.hero-section {
  background: linear-gradient(to right, #fe4f4f 0%, #93929261 100%);
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  padding: 20px;
}

/* Title styling */
.title {
  font-size: 3rem;
  font-weight: 700;
  color: #ffffff;
  margin-bottom: 20px;
}

/* Intro text styling */
.intro-text {
  font-size: 1.2rem;
  color: #e0f7fa;
  line-height: 1.6;
  max-width: 600px;
  margin: 0 auto 30px;
  padding: 0 15px;
}

/* CTA button styling */
.cta-button {
  background-color: #ff6b6b;
  color: #ffffff;
  padding: 12px 30px;
  font-size: 1.1rem;
  font-weight: 600;
  border-radius: 25px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
  transition: background-color 0.3s ease, box-shadow 0.3s ease;
}

.cta-button:hover {
  background-color: #ee5253;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}
</style>
