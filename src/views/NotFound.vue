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