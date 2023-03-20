import { createRouter, createWebHistory } from 'vue-router';
import Home from '../views/Home.vue';
import Viz from '../views/Viz.vue';
import Model from '../views/Model.vue';
const routes = [
    {
      path: '/',
      name: 'Home',
      component: Home
    },
    {
      path: '/home',
      name: 'Home',
        component: Home,
        meta: {
            title: 'Home'
        }
    },
    {
      path: '/Viz',
      name: 'Viz',
      component: Viz,
      meta: {
        title: 'Viz'
      }
    },
    {
      path: '/model',
      name: 'Model',
      component: Model,
      meta: {
        title: 'Model'
      }
    }
  ];
  
const router = createRouter({
    history: createWebHistory(process.env.BASE_URL),
    routes
});
  
router.beforeEach((to, from, next) => {
    document.title = to.meta.title ? to.meta.title + '|' + 'Vue 3' : 'Vue 3';
    next();
});
  
export default router;