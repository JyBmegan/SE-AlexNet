const modelStates = {
  input: {
    image: "assets/faces/full-15.jpg",
    alt: "Full face stimulus",
    kicker: "Stimulus",
    title: "Start with the same face",
    caption: "A 224 by 224 aligned full-face stimulus used for model probing.",
    text:
      "The aligned input makes attention the visual variable."
  },
  alexnet: {
    image: "assets/heatmaps/alexnet-face-full-1.png",
    alt: "AlexNet FaceBased Grad-CAM heatmap",
    kicker: "Baseline AlexNet",
    title: "Baseline attention gives the reference point",
    caption: "Grad-CAM heatmap from the FaceBased AlexNet baseline.",
    text:
      "Baseline AlexNet sets the reference for the SE comparison."
  },
  vgg16: {
    image: "assets/heatmaps/vgg16-face-full-1.png",
    alt: "VGG16 FaceBased Grad-CAM heatmap",
    kicker: "VGG16 baseline",
    title: "A deeper benchmark checks that the effect is not a shallow-model artifact",
    caption: "Grad-CAM heatmap from the FaceBased VGG16 baseline.",
    text:
      "VGG16 separates model depth from SE-based reweighting."
  },
  sealexnet: {
    image: "assets/heatmaps/sealexnet-eyes-mouth-full-15.png",
    alt: "SE-AlexNet FaceBased Grad-CAM heatmap with eyes and mouth attention",
    kicker: "SE-AlexNet Location-3 R16",
    title: "Location-3 concentrates evidence differently",
    caption: "Grad-CAM heatmap from a Location-3 FaceBased R16 condition, selected for visible eye and mouth evidence.",
    text:
      "The Location-3 model changes where visual evidence concentrates."
  }
};

const roiStats = {
  Full: {
    label: "Full-face stimuli",
    values: [
      ["Eyes", 0.21261411485714285, "#2bdad7"],
      ["Nose", 0.008528178, "#f0cf3f"],
      ["Mouth", 0.033296798380952376, "#f16666"]
    ],
    note: "Face-pretrained Location-3/R16 full-face stimuli show the largest mean share in the eyes ROI."
  },
  E: {
    label: "Eyes-masked stimuli",
    values: [
      ["Eyes", 0.02796001776190476, "#2bdad7"],
      ["Nose", 0.030657087, "#f0cf3f"],
      ["Mouth", 0.03568825842857143, "#f16666"]
    ],
    note: "For eye-masked stimuli, the mean ROI shares are relatively even across the three predefined bands."
  },
  M: {
    label: "Mouth-masked stimuli",
    values: [
      ["Eyes", 0.2663587358095238, "#2bdad7"],
      ["Nose", 0.06051111571428571, "#f0cf3f"],
      ["Mouth", 0.0028780205238095238, "#f16666"]
    ],
    note: "For mouth-masked stimuli, the mean share is concentrated in the eyes ROI."
  },
  N: {
    label: "Nose-masked stimuli",
    values: [
      ["Eyes", 0.004540437380952381, "#2bdad7"],
      ["Nose", 0.0, "#f0cf3f"],
      ["Mouth", 0.0, "#f16666"]
    ],
    note: "For nose-masked stimuli, the current paper-matched input has near-zero mean values in all three predefined ROI bands."
  }
};

const tabs = document.querySelectorAll(".compare-tab");
const modelImage = document.querySelector("#modelImage");
const modelCaption = document.querySelector("#modelCaption");
const modelKicker = document.querySelector("#modelKicker");
const modelTitle = document.querySelector("#modelTitle");
const modelText = document.querySelector("#modelText");
const modelOrder = ["input", "alexnet", "vgg16", "sealexnet"];
let activeModelIndex = 0;
let modelAutoplayTimer;

function activateModel(modelKey) {
  const state = modelStates[modelKey];
  if (!state) return;

  tabs.forEach((item) => {
    const isActive = item.dataset.model === modelKey;
    item.classList.toggle("is-active", isActive);
    item.setAttribute("aria-selected", isActive ? "true" : "false");
  });

  modelImage.classList.add("is-fading");
  window.setTimeout(() => {
    modelImage.src = state.image;
    modelImage.alt = state.alt;
    modelCaption.textContent = state.caption;
    if (modelKicker) {
      modelKicker.textContent = state.kicker;
    }
    modelTitle.textContent = state.title;
    modelText.textContent = state.text;
    modelImage.classList.remove("is-fading");
  }, 120);

  activeModelIndex = modelOrder.indexOf(modelKey);
}

function advanceModel() {
  activeModelIndex = (activeModelIndex + 1) % modelOrder.length;
  activateModel(modelOrder[activeModelIndex]);
}

function stopModelAutoplay() {
  window.clearInterval(modelAutoplayTimer);
}

function startModelAutoplay() {
  stopModelAutoplay();
  modelAutoplayTimer = window.setInterval(advanceModel, 3000);
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    activateModel(tab.dataset.model);
    startModelAutoplay();
  });
});

if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
  startModelAutoplay();
}

const conditionTabs = document.querySelectorAll(".condition-tab");
const barChart = document.querySelector("#barChart");
const chartNote = document.querySelector("#chartNote");

function renderChart(condition) {
  const state = roiStats[condition];
  const sortedValues = [...state.values].sort((a, b) => b[1] - a[1]);
  const maxValue = sortedValues[0][1];
  barChart.innerHTML = "";

  sortedValues.forEach(([label, value, color]) => {
    const row = document.createElement("div");
    row.className = "rank-row";

    const rowLabel = document.createElement("div");
    rowLabel.className = "rank-label";
    rowLabel.textContent = label;

    const track = document.createElement("div");
    track.className = "rank-track";

    const fill = document.createElement("div");
    fill.className = "rank-fill";
    fill.style.background = color;
    fill.style.width = `${Math.max(18, (value / maxValue) * 100)}%`;

    const valueLabel = document.createElement("div");
    valueLabel.className = "rank-value";
    valueLabel.textContent = `${(value * 100).toFixed(1)}%`;

    track.appendChild(fill);
    row.append(rowLabel, track, valueLabel);
    barChart.appendChild(row);
  });

  chartNote.textContent = state.note;
}

conditionTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    conditionTabs.forEach((item) => item.classList.remove("is-active"));
    tab.classList.add("is-active");
    renderChart(tab.dataset.condition);
  });
});

renderChart("Full");
