<template>
  <div class="wrap">
    <div class="row">
      <label class="label">{{ label }}</label>
      <select class="select" v-model="selected">
        <option value="">请选择</option>
        <option v-for="it in items" :key="it.path" :value="it.path">
          {{ it.plate }} / {{ it.well }} / {{ basename(it.path) }}
        </option>
      </select>
    </div>
    <div class="path" v-if="selected">{{ selected }}</div>
  </div>
</template>

<script setup>
import { onMounted, ref, watch } from "vue";

const props = defineProps({
  label: { type: String, default: "Image" },
  modelValue: { type: String, default: "" },
  apiBase: { type: String, default: "http://localhost:8787" },
});

const emit = defineEmits(["update:modelValue"]);

const items = ref([]);
const selected = ref(props.modelValue || "");

watch(
  () => props.modelValue,
  (v) => {
    if (typeof v === "string" && v !== selected.value) {
      selected.value = v;
    }
  }
);

watch(selected, (v) => emit("update:modelValue", v));

function basename(p) {
  const s = String(p || "");
  const i = s.lastIndexOf("/");
  return i >= 0 ? s.slice(i + 1) : s;
}

async function load() {
  const base = String(props.apiBase || "").replace(/\/$/, "");
  const u = `${base}/api/images`;
  const r = await fetch(u);
  const j = await r.json();
  const xs = j && Array.isArray(j.items) ? j.items : [];
  items.value = xs;
}

onMounted(() => {
  load();
});
</script>

<style scoped>
.wrap {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.row {
  display: flex;
  align-items: center;
  gap: 12px;
}
.label {
  min-width: 90px;
  font-weight: 600;
}
.select {
  flex: 1;
  padding: 8px 10px;
}
.path {
  font-size: 12px;
  opacity: 0.8;
  word-break: break-all;
}
</style>
