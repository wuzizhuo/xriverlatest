<template>
  <div class="page" :style="{ backgroundImage: `url(${bgUrl})` }">
    <div class="overlay">
      <div class="title">Virtual Cell</div>
      <div class="actions">
        <button class="btn" @click="go('/growthfactor')">growthfactor</button>
        <button class="btn" @click="go('/growthprocess')">growthprocess</button>
        <button class="btn" @click="go('/response')">Response</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { useRouter } from "vue-router";

const router = useRouter();
const bgUrl = new URL("../素材/首页.png", import.meta.url).href;

function go(p) {
  router.push(p);
}
</script>

<style scoped>
.page {
  min-height: 100vh;
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  display: flex;
  align-items: center;
  justify-content: center;
}
.overlay {
  width: min(900px, 92vw);
  padding: 28px;
  background: rgba(0, 0, 0, 0.55);
  color: #fff;
  border-radius: 12px;
  backdrop-filter: blur(4px);
}
.title {
  font-size: 34px;
  font-weight: 700;
  letter-spacing: 1px;
  margin-bottom: 22px;
}
.actions {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
}
.btn {
  padding: 12px 18px;
  font-size: 16px;
  border: 1px solid rgba(255, 255, 255, 0.35);
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
  cursor: pointer;
  border-radius: 10px;
}
.btn:hover {
  background: rgba(255, 255, 255, 0.2);
}
</style>
