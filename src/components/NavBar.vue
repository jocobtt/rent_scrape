<template>
  <header>
    <nav class="navbar">
      <div class="container mx-auto flex items-center justify-between">
        <a class="navbar-brand" href="/">Rent Predictor</a>
        <div class="navbar-links hidden md:flex ml-auto">
          <ul class="navbar-nav flex items-center space-x-6">
            <li v-for="link in navLinks" :key="link.to" class="nav-item">
              <router-link class="nav-link" :to="link.to">{{ link.text }}</router-link>
            </li>
          </ul>
        </div>
        <!-- Mobile menu toggle -->
        <button @click="toggleMobileMenu" class="md:hidden text-white" aria-label="Toggle mobile menu">
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" class="w-6 h-6">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16m-7 6h7"/>
          </svg>
        </button>
      </div>
      <!-- Mobile menu -->
      <div v-if="isMobileMenuOpen" class="mobile-menu md:hidden">
        <ul class="navbar-nav flex flex-col items-center space-y-4">
          <li v-for="link in navLinks" :key="link.to" class="nav-item">
            <router-link class="nav-link" :to="link.to">{{ link.text }}</router-link>
          </li>
        </ul>
      </div>
    </nav>
    <div class="main-content">
      <!-- Your other content goes here -->
    </div>
  </header>
</template>

<script>
import { defineComponent, ref } from 'vue';

export default defineComponent({
  name: 'NavBar',
  setup() {
    const isMobileMenuOpen = ref(false);
    const navLinks = [
      { to: '/home', text: 'Home' },
      { to: '/viz', text: 'Visualize Rent' },
      { to: '/model', text: 'Calculate Rent' },
      { to: '/about', text: 'About' }
    ];

    function toggleMobileMenu() {
      isMobileMenuOpen.value = !isMobileMenuOpen.value;
    }

    return { isMobileMenuOpen, toggleMobileMenu, navLinks };
  }
});
</script>

<style scoped>
/* Global styles */
body, html {
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  font-family: 'Roboto', sans-serif;
}

/* Navbar styling */
.navbar {
  background-color: #1e293b; /* Dark blue-gray */
  padding: 10px 20px;
  width: 100%;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 1000;
}

.navbar-brand {
  color: #ffffff;
  font-size: 1.8rem;
  font-weight: 700;
  text-decoration: none;
  transition: color 0.3s ease;
}

.navbar-brand:hover {
  color: #4facfe; /* Accent color */
}

.navbar-nav {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  align-items: center;
}


.nav-item {
  margin-left: 20px;
}

.nav-link {
  color: #ffffff;
  font-size: 1rem;
  padding: 8px 20px;
  border-radius: 5px;
  text-decoration: none;
  transition: background-color 0.3s ease, color 0.3s ease;
}

.nav-link:hover {
  background-color: #4b5563; /* Slightly darker for hover */
  color: #ffffff;
}

/* Mobile menu */
.mobile-menu {
  background-color: #1e293b;
  padding: 20px;
  border-top: 1px solid #4b5563;
}

/* Main content padding */
.main-content {
  padding-top: 70px; /* Adjust this value as needed */
}

/* Mobile menu button */
button {
  background: none;
  border: none;
  cursor: pointer;
  outline: none;
}
</style>

