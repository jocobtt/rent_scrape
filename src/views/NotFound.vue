<template>
    <div class="flex items-center justify-center h-screen">
        <div class="text-4xl font-bold">
        <h1 class="text-4xl font-bold">404 - Page not found</h1>
        <p class="text-2xl">Whoops! The page you are looking for does not exist.</p>
        <router-link to="/" class="text-blue-500">Go back home</router-link>
        </div>
    </div>
</template>


<script>
export default {
  name: "NotFound",
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