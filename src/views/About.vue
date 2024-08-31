<template>
  <main class="container mx-auto px-6 py-8 max-w-3xl">
    <section aria-labelledby="about-project">
      <header class="mb-6">
        <h1 id="about-project" class="text-4xl font-bold text-gray-800 tracking-tight">About this Project</h1>
      </header>

      <article class="text-gray-700 leading-relaxed">
        <p class="mb-6">
          This project is a Vue/Vite-based application with a FastAPI backend hosted on Vercel. The data is scraped from Suumo, a Japanese apartment listing site. We build our models using a few different methods; we have one that compares different regression-based models and another that is a boosting-based regression model. Both of these help us predict the rent of an apartment based on its features.
        </p>
        <p>
          We are generating the two models to help us predict a person's rent based on the apartment they are looking at. Additionally, we are visualizing some parts of the data itself. One of the models is a challenger model, and another is our baseline model. We are using the challenger model to see if we can improve on the baseline model. D3 is used for visualizing the data.
        </p>
      </article>
    </section>
  </main>
</template>

<script>
export default {
  name: "AboutView",
};
</script>

<style scoped>
/* Custom styles for additional fine-tuning if needed */
.container {
  max-width: 800px; /* This is already set by Tailwind's max-w-3xl class */
}

/* Example of adding a transition effect to text */
h1 {
  transition: color 0.3s;
}

h1:hover {
  color: #1a202c; /* Change color on hover for interactivity */
}
</style>