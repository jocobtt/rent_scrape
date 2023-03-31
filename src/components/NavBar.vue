<template>
    <header>
      <nav class="navbar navbar-dark bg-dark">
        <div class="container">
          <a class="navbar-brand" href="/">Baby Name Generator</a>
          <div class="collapse navbar-collapse" id="navbarCollapse">
            <ul class="navbar-nav me-auto mb-2 mb-md-0">
              <li class="nav-item">
                <router-link class="nav-link" to="/home">Home</router-link>
              </li>
              <li class="nav-item">
                <router-link class="nav-link" to="/viz">Visualize rent prices</router-link>
              </li>
              <li class="nav-item">
                <router-link class="nav-link" to="/about">About</router-link>
              </li>
              <li class="nav-item">
                <router-link class="nav-link" to="/bar">BarChart</router-link>
              </li>
              <!-- <li class="nav-item">
                <router-link class="nav-link" to="/model">Calculate Rent</router-link>
              </li> -->
            </ul>
          </div>
        </div>
      </nav>
    </header>
  </template>
  
  <script>
  import { defineComponent } from 'vue';
  import { useRouter } from 'vue-router';
  const router = useRouter();
  export default defineComponent({
    name: 'NavBar',
  });
  </script>
  
  <style scoped>
  a {
    cursor: pointer;
  }
  </style>