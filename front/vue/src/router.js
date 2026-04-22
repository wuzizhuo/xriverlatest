

import { createRouter, createWebHashHistory } from "vue-router";

import Total from "../total.vue";
import GrowthFactor from "../growth_factor.vue";
import GrowthProcess from "../grwoth_process.vue";
import ResponsePage from "../Respons.vue";

export const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: "/", component: Total },
    { path: "/growthfactor", component: GrowthFactor },
    { path: "/growthprocess", component: GrowthProcess },
    { path: "/response", component: ResponsePage },
  ],
});

