<template>
  <div class="page">
    <div class="header">
      <div class="h1">growth factor</div>
      <button class="link" @click="goHome">返回首页</button>
    </div>

    <div class="card">
      <ImageSelect label="source image" v-model="sourceImage" :apiBase="apiBase" />
      <ImageSelect label="target image" v-model="targetImage" :apiBase="apiBase" />

      <div class="row">
        <button class="btn" @click="submit">确认选择</button>
        <div class="status" v-if="status">{{ status }}</div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";
import { useRouter } from "vue-router";
import ImageSelect from "./src/components/ImageSelect.vue";

const router = useRouter();
const apiBase = "http://localhost:8787";
const sourceImage = ref("");
const targetImage = ref("");
const status = ref("");

function goHome() {
  router.push("/");
}

async function submit() {
  status.value = "";
  const u = `${apiBase.replace(/\/$/, "")}/api/selection`;
  const body = { module: "growth_factor", source_image: sourceImage.value, target_image: targetImage.value };
  const r = await fetch(u, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json();
  status.value = `saved ${j.updated_at || ""}`;
}
</script>

<style scoped>
.page {
  min-height: 100vh;
  padding: 22px;
  background: #0b0f17;
  color: #e7eefc;
}
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}
.h1 {
  font-size: 22px;
  font-weight: 700;
}
.link {
  padding: 8px 12px;
  background: transparent;
  border: 1px solid rgba(231, 238, 252, 0.2);
  color: #e7eefc;
  border-radius: 10px;
  cursor: pointer;
}
.card {
  max-width: 980px;
  padding: 18px;
  border: 1px solid rgba(231, 238, 252, 0.12);
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.03);
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.row {
  display: flex;
  align-items: center;
  gap: 14px;
}
.btn {
  padding: 10px 14px;
  border-radius: 10px;
  border: 1px solid rgba(231, 238, 252, 0.2);
  background: rgba(255, 255, 255, 0.08);
  color: #e7eefc;
  cursor: pointer;
}
.status {
  font-size: 12px;
  opacity: 0.85;
}
</style>
