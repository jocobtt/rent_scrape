<template>
  <div class="not-found">
    <div class="not-found__container">
      <h1 class="not-found__title">404 - Page Not Found</h1>
      <p class="not-found__message">Whoops! The page you are looking for does not exist.</p>
      <router-link to="/" class="not-found__link">Go back home</router-link>
    </div>
  </div>
</template>

<script>
export default {
  name: "NotFound",
};
</script>

<style scoped>
.not-found {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background-color: #f8f9fa; /* Light grey background */
  color: #333; /* Consistent dark text color */
  font-family: 'IBM Plex Mono', monospace; /* Using the IBM Plex Mono font */
}

.not-found__container {
  text-align: center;
  background-color: #fff;
  padding: 40px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Light shadow for depth */
  border-radius: 8px;
}

.not-found__title {
  font-size: 2.5rem;
  margin-bottom: 20px;
  color: #ff0000; /* Highlight error in red */
}

.not-found__message {
  font-size: 1.25rem;
  margin-bottom: 20px;
  color: #666; /* Subtle grey color for message */
}

.not-found__link {
  font-size: 1rem;
  color: #007bff;
  text-decoration: none;
  border-bottom: 1px solid #007bff; /* Underline link */
  transition: color 0.3s;
}

.not-found__link:hover {
  color: #0056b3; /* Darker blue on hover */
  border-bottom: 1px solid #0056b3; /* Darker underline on hover */
}
</style>