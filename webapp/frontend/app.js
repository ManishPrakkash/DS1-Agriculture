const API = "";

function el(tag, className, html) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

// --- tab switching -----------------------------------------------------
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

let META = null;

async function loadMeta() {
  const res = await fetch(`${API}/api/meta`);
  if (!res.ok) throw new Error(`Failed to load /api/meta (${res.status})`);
  META = await res.json();

  document.getElementById("hero-r2").textContent = `${(META.model.test_r2 * 100).toFixed(1)}%`;

  renderStatCards(META);
  renderCharts(META);
  renderInsights(META);
  renderActionPlan(META);
  populatePredictForm(META);
}

const STAT_ICONS = ["\u{1F4CA}", "\u{1F30D}", "\u{1F33E}", "\u{1F4C5}", "\u{1F3AF}", "\u{1F4CF}", "\u{1F4D0}"];

function renderStatCards(meta) {
  const wrap = document.getElementById("stat-cards");
  const cards = [
    { label: "Observations (country × crop × year)", value: meta.dataset.n_rows.toLocaleString() },
    { label: "Countries", value: meta.dataset.n_countries },
    { label: "Crop types", value: meta.dataset.n_crops },
    { label: "Years covered", value: `${meta.dataset.year_min}–${meta.dataset.year_max}` },
    { label: "Model test R²", value: meta.model.test_r2 },
    { label: "Model test MAE (kg/ha)", value: meta.model.test_mae.toLocaleString() },
    { label: "Model test RMSE (kg/ha)", value: meta.model.test_rmse.toLocaleString() },
  ];
  wrap.innerHTML = "";
  cards.forEach((c, i) => {
    const card = el("div", "stat-card");
    card.appendChild(el("span", "ic", STAT_ICONS[i] || "•"));
    card.appendChild(el("div", "value", c.value));
    card.appendChild(el("div", "label", c.label));
    wrap.appendChild(card);
  });
}

function renderCharts(meta) {
  const wrap = document.getElementById("charts");
  wrap.innerHTML = "";
  meta.charts.forEach((c, i) => {
    const card = el("div", "chart-card");
    card.appendChild(el("span", "fig-tag", `FIG. ${i + 1}`));
    const img = el("img");
    img.src = `/charts/${c.file}`;
    img.alt = c.caption;
    img.loading = "lazy";
    card.appendChild(img);
    card.appendChild(el("p", "", c.caption));
    wrap.appendChild(card);
  });
}

function renderInsights(meta) {
  const wrap = document.getElementById("insights");
  wrap.innerHTML = "";
  meta.insights.forEach((ins, i) => {
    const card = el("div", "insight-card");
    card.appendChild(el("div", "insight-num", String(i + 1)));
    const body = el("div");
    body.appendChild(el("h3", "", ins.title));
    body.appendChild(el("p", "ev", ins.evidence));
    if (ins.limitation) body.appendChild(el("p", "lim", ins.limitation));
    card.appendChild(body);
    wrap.appendChild(card);
  });
}

function renderActionPlan(meta) {
  const tbody = document.querySelector("#action-plan tbody");
  tbody.innerHTML = "";
  meta.action_plan.forEach((row) => {
    const tr = el("tr");
    [row.evidence, row.action, row.purpose, row.limitation].forEach((txt) => {
      tr.appendChild(el("td", "", txt));
    });
    tbody.appendChild(tr);
  });
}

function populatePredictForm(meta) {
  const countrySel = document.getElementById("f-country");
  const cropSel = document.getElementById("f-crop");
  countrySel.innerHTML = meta.countries.map((c) => `<option value="${c}">${c}</option>`).join("");
  cropSel.innerHTML = meta.crops.map((c) => `<option value="${c}">${c}</option>`).join("");

  const r = meta.ranges;
  document.getElementById("f-year").value = r.Year.max;
  document.getElementById("f-year").min = r.Year.min;
  document.getElementById("f-year").max = r.Year.max + 5;
  document.getElementById("f-fertiliser").value = r.Fertiliser_Use_kg_ha.mean;
  document.getElementById("f-irrigation").value = r.Irrigation_pct.mean;
  document.getElementById("f-temperature").value = r.Temperature_C.mean;
  document.getElementById("f-rainfall").value = r.Rainfall_mm.mean;
}

function buildGauge(predicted, band, yieldRange) {
  const min = yieldRange.min;
  const max = yieldRange.max;
  const span = max - min || 1;
  const pct = (v) => Math.min(100, Math.max(0, ((v - min) / span) * 100));

  const wrap = el("div", "gauge-wrap");
  const track = el("div", "gauge-track");

  const bandEl = el("div", "gauge-band");
  bandEl.style.left = `${pct(band[0])}%`;
  bandEl.style.width = `${Math.max(1, pct(band[1]) - pct(band[0]))}%`;
  track.appendChild(bandEl);

  const marker = el("div", "gauge-marker");
  marker.style.left = `${pct(predicted)}%`;
  marker.title = `Predicted: ${Math.round(predicted).toLocaleString()} kg/ha`;
  track.appendChild(marker);

  wrap.appendChild(track);
  const labels = el("div", "gauge-labels");
  labels.appendChild(el("span", "", `${Math.round(min).toLocaleString()} kg/ha`));
  labels.appendChild(el("span", "", "range for this crop"));
  labels.appendChild(el("span", "", `${Math.round(max).toLocaleString()} kg/ha`));
  wrap.appendChild(labels);
  return wrap;
}

document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const resultBox = document.getElementById("predict-result");
  resultBox.innerHTML = "";
  const loadingWrap = el("div", "placeholder-wrap");
  loadingWrap.appendChild(el("span", "placeholder-ic", "⏳"));
  loadingWrap.appendChild(el("p", "placeholder", "Running the Random Forest model..."));
  resultBox.appendChild(loadingWrap);

  const payload = {
    country: document.getElementById("f-country").value,
    crop_type: document.getElementById("f-crop").value,
    year: parseInt(document.getElementById("f-year").value, 10),
    fertiliser_use_kg_ha: parseFloat(document.getElementById("f-fertiliser").value),
    irrigation_pct: parseFloat(document.getElementById("f-irrigation").value),
    temperature_c: parseFloat(document.getElementById("f-temperature").value),
    rainfall_mm: parseFloat(document.getElementById("f-rainfall").value),
  };

  try {
    const res = await fetch(`${API}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Prediction failed");

    resultBox.innerHTML = "";
    resultBox.appendChild(el("p", "eyebrow", `${payload.crop_type} · ${payload.country} · ${payload.year}`));
    resultBox.appendChild(el("p", "yield-big", `${data.predicted_yield_kg_ha.toLocaleString()} kg/ha`));
    resultBox.appendChild(el("p", "yield-sub", `≈ ${data.predicted_yield_tonnes_ha} t/ha`));

    const cropRange = META && META.crop_yield_ranges && META.crop_yield_ranges[payload.crop_type];
    if (cropRange) {
      resultBox.appendChild(buildGauge(data.predicted_yield_kg_ha, data.uncertainty_band_kg_ha, cropRange));
    }
    resultBox.appendChild(el("p", "yield-sub", `80% of the forest's trees predict between ${data.uncertainty_band_kg_ha[0].toLocaleString()} and ${data.uncertainty_band_kg_ha[1].toLocaleString()} kg/ha (shaded band above).`));
    resultBox.appendChild(el("div", "note", data.note));
  } catch (err) {
    resultBox.innerHTML = "";
    const errWrap = el("div", "placeholder-wrap");
    errWrap.appendChild(el("span", "placeholder-ic", "⚠️"));
    errWrap.appendChild(el("p", "error", err.message));
    resultBox.appendChild(errWrap);
  }
});

loadMeta().catch((err) => {
  document.getElementById("stat-cards").innerHTML = `<p class="error">Could not load dashboard data: ${err.message}. Is the backend running?</p>`;
});
