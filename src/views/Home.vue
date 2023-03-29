<template>
    <div id="home-page">
      <h1>Predict your rent!</h1>
      <div id="main-part">
        <p> This App showcases a few different ways to generate potential names for a baby. Try it out in the different pages we have available and please give any feedback you have, if any!</p>
        <br />
        <p> The first model is a randomly generated name using a Large Language Model! If you want to learn more go to the about page.</p>
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
    color: #2c3e50;
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
    background: #eee;
  }
  .buttons {
    margin-top: 2rem;
  }
  </style>