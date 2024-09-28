import { createRouter, createWebHistory } from 'vue-router';

const routes = [
  {
    path: '/',
    redirect: '/home'
  },
  {
      path: '/home',
      name: 'Home',
        component: () => import('../views/Home.vue'),
        meta: {
            title: 'Home'
        }
    },
    {
      path: '/Viz',
      name: 'Viz',
      component: () => import('../views/Viz.vue'),
      meta: {
        title: 'Viz'
      }
    },
    {
      path: '/model',
      name: 'Model',
      component: () => import('../views/Model.vue'),
      meta: {
        title: 'Model'
      }
    },
    {
      path: '/about',
      name: 'About',
      component: () => import('../views/About.vue'),
      meta: {
        title: 'About'
      }
    },
    {
      path: '/barChart',
      name: 'BarChart',
      component: () => import('../views/BarChartView.vue'),
      meta: {
        title: 'BarChart'
      }
    },
    {
      path: '/scatterPlot',
      name: 'ScatterPlot',
      component: () => import('../views/ScatterPlotView.vue'),
      meta: {
        title: 'ScatterPlot'
      }
    },
    {
      path: '/:pathMatch(.*)*',
      name: 'NotFound',
      component: () => import('../views/NotFound.vue'),
      meta: {
        title: '404'
      }
    }
  ];
  
const router = createRouter({
    history: createWebHistory(),
    routes
});
  
router.beforeEach((to, from, next) => {
    document.title = to.meta.title ? to.meta.title + '|' + 'Vue 3' : 'Vue 3';
    next();
});
  
export default router;