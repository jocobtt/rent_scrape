<template>
  <main class="container mx-auto px-6 py-8 max-w-5xl bg-gradient-to-br from-gray-50 to-gray-100 min-h-screen">
    <!-- Header Section -->
    <header class="mb-8 text-center">
      <h2 class="text-3xl font-semibold text-gray-800 mb-2">About Our Project</h2>
      <h1 class="text-4xl sm:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-600 tracking-tight">
        Predicting Apartment Rents in Japan with Machine Learning
      </h1>
    </header>

    <!-- Project Description Section -->
    <article class="text-gray-800 leading-relaxed space-y-6">
      <p class="text-lg sm:text-xl font-light bg-white shadow-lg p-6 rounded-lg">
        Welcome to our Vue/Vite application with a FastAPI backend, hosted on Vercel. We’ve developed an innovative solution for predicting apartment rents in Japan, leveraging data from
        <a href="https://suumo.jp/" target="_blank" rel="noopener noreferrer" class="custom-link">Suumo</a>, a leading Japanese real estate platform.
      </p>

      <!-- Machine Learning Models Section -->
      <div class="text-lg sm:text-xl bg-white shadow-lg p-6 rounded-lg space-y-4">
        <p>Our project showcases two sophisticated machine learning models:</p>
        <ul class="list-disc list-inside pl-6 text-gray-700 space-y-2">
          <li>Comparative analysis of various regression-based models</li>
          <li>Advanced boosting-based regression model</li>
        </ul>
      </div>

      <!-- Visualization & Challenger Model Section -->
      <p class="text-lg sm:text-xl bg-white shadow-lg p-6 rounded-lg">
        These models provide accurate rent predictions based on apartment features, offering valuable insights for both renters and property managers. We’ve implemented a challenger model approach, continuously improving our baseline predictions. Our interactive data visualizations, powered by D3.js, offer an engaging and intuitive way to explore Japanese rental market trends.
        You can find the source code for this project's backend <a href="https://github.com/jocobtt/rent_scrape" target="_blank" rel="noopener noreferrer" class="custom-link">here</a> on GitHub. 
        And the frontend code is available <a href="https://github.com/jocobtt/rent_d3" target="_blank" rel="noopener noreferrer" class="custom-link">here</a>.
      </p>
    </article>

    <!-- Technology Stack Section -->
    <section class="mt-8">
      <h2 class="text-3xl sm:text-4xl font-bold text-gray-800 text-center mb-8">
        Our Technology Stack
      </h2>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        <div v-for="(technology, index) in technologies" :key="index" 
             class="bg-white shadow-lg p-6 rounded-lg hover:shadow-xl transform transition-all duration-300 hover:-translate-y-2 hover:bg-gray-50">
          <div class="flex items-center mb-4">
            <img :src="technology.icon" alt="" class="w-10 h-10 mr-4" />
            <h3 class="text-2xl font-semibold text-gray-800">{{ technology.name }}</h3>
          </div>
          <p class="text-gray-600">{{ technology.description }}</p>
        </div>
      </div>
    </section>
  </main>
</template>

<script>
export default {
  name: "AboutView",
  data() {
    return {
      technologies: [
        { name: "Vue.js", description: "A progressive JavaScript framework for building dynamic user interfaces.", icon: "/path-to-icon/vue.svg" },
        { name: "Vite", description: "Next-generation frontend tooling for faster and leaner web development.", icon: "/path-to-icon/vite.svg" },
        { name: "FastAPI", description: "High-performance Python framework for building modern APIs.", icon: "/path-to-icon/fastapi.svg" },
        { name: "D3.js", description: "Powerful library for creating interactive data visualizations.", icon: "/path-to-icon/d3.svg" },
        { name: "Vercel", description: "Cloud platform for seamless deployment and scaling.", icon: "/path-to-icon/vercel.svg" },
        { name: "Tailwind CSS", description: "Utility-first CSS framework for rapid UI development.", icon: "/path-to-icon/tailwind.svg" },
      ],
    };
  },
};
</script>

<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

body, html {
  font-family: 'Inter', sans-serif;
  background-color: #f9fafb;
  color: #1a202c;
}

h1, h2 {
  transition: transform 0.3s ease-in-out;
}

h1:hover, h2:hover {
  transform: scale(1.02);
}

article p, article div {
  font-size: 1.125rem;
  line-height: 1.8;
}


ul.list-disc {
  margin-top: 10px;
  padding-left: 15px;
}

ul.list-disc li {
  font-size: 1rem;
}

.grid {
  gap: 1.5rem;
  padding: 0 10px;
}

.custom-link {
  color: #1a73e8; 
  text-decoration: underline; 
  font-weight: bold; 
}

.custom-link:hover {
  color: #0c47a1; 
  text-decoration: none; 
}

/* Technology Stack Card Hover Effect */
.hover:hover {
  background-color: #f0f4f8;
}

@media (max-width: 768px) {
  h1 {
    font-size: 2.75rem;
  }
  
  h2 {
    font-size: 2.25rem;
  }

  article p {
    font-size: 1rem;
  }
}
</style>


