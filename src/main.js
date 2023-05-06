import { createApp } from 'vue'
import './style.css'
import App from './App.vue'
import router from './router'

// add middleware
// import axios from 'axios'
// axios.defaults.withCredentials = true;
// axios.defaults.baseURL = 'http://localhost:3000'; // replace to cloud run api


// add vuetify 
import "vuetify/styles";
import { createVuetify } from "vuetify";
import * as components from "vuetify/components";
import * as directives from "vuetify/directives";

const vuetify = createVuetify({
    components,
    directives,
});

// axios.defaults.withCredentials = true;
// axios.defaults.baseURL = 'http://localhost:3000'; // replace to cloud run api

createApp(App).use(router).use(vuetify).mount('#app');
