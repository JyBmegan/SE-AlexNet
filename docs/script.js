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
    title: "Late SE insertion concentrates evidence differently",
    caption: "Grad-CAM heatmap from a Location-3 FaceBased R16 condition, selected for visible eye and mouth evidence.",
    text:
      "Late SE insertion changes where visual evidence concentrates."
  }
};

const roiStats = {
  Full: {
    label: "Full-face stimuli",
    values: [
      ["Eyes", 0.0991988655052579, "#2bdad7"],
      ["Nose", 0.09499713399226689, "#f0cf3f"],
      ["Mouth", 0.08259077352165407, "#f16666"]
    ],
    note: "Full faces distribute saliency across eyes, nose, and mouth, with eyes slightly higher in the aggregate ROI summary."
  },
  E: {
    label: "Eyes-masked stimuli",
    values: [
      ["Eyes", 0.06356021169603061, "#2bdad7"],
      ["Nose", 0.1002952228523284, "#f0cf3f"],
      ["Mouth", 0.08744287633679274, "#f16666"]
    ],
    note: "When the eye region is masked, saliency shifts toward remaining informative regions, especially nose and mouth."
  },
  M: {
    label: "Mouth-masked stimuli",
    values: [
      ["Eyes", 0.11924139726359918, "#2bdad7"],
      ["Nose", 0.08828131345537334, "#f0cf3f"],
      ["Mouth", 0.06523406982112628, "#f16666"]
    ],
    note: "When the mouth is masked, average saliency rises in the eye region."
  },
  N: {
    label: "Nose-masked stimuli",
    values: [
      ["Eyes", 0.112346980862488, "#2bdad7"],
      ["Nose", 0.06741387068382867, "#f0cf3f"],
      ["Mouth", 0.09159348543430441, "#f16666"]
    ],
    note: "When the nose is masked, saliency is carried more by eyes and mouth than by the unavailable nose band."
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
