<template>
  <v-container id="home-page">
    <v-row justify="center">
      <v-col cols="12" sm="8" md="6">
        <h1>Predict your rent! 🗼</h1>
        <div id="main-part">
          <p>
            This app is for visualizing and predicting rent in the greater Tokyo area!
          </p>
          <!-- change this to instead be a directory of sorts and intro -->
        </div>
      </v-col>
    </v-row>
  </v-container>
</template>
  
<script>
export default {
  name: "HomeView",
};
</script>
  
<style>
#home-page {
  min-height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}
</style>