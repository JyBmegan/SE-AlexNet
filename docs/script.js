const controls = [...document.querySelectorAll('.control-set button')];
const grid = document.querySelector('#comparison-grid');
const label = document.querySelector('#explorer-label');
const source = 'assets/paper/figure7-gradcam.jpeg';
const groups = {early:117, middle:676, late:1227, alex:1830, vgg:1985};
const ratios = [2,4,8,16,32];
const selection = {pretraining:'face',model:'late',reduction:16};
const conditions = [
  {name:'Full face', input:'assets/faces/full-11.jpg'},
  {name:'Eyes masked', input:'assets/faces/eyes-masked-11.jpg'},
  {name:'Nose masked', input:'assets/faces/nose-masked-11.jpg'},
  {name:'Mouth masked', input:'assets/faces/mouth-masked-11.jpg'}
];
function makeHeatmap(x,y,alt) {
  const crop = document.createElement('div');
  crop.className = 'paper-crop';
  const image = document.createElement('img');
  image.className = 'paper-crop-image';
  image.src = source;
  image.alt = alt;
  image.width = 2125;
  image.height = 1027;
  image.loading = 'lazy';
  image.style.left = `${-x / 101 * 100}%`;
  image.style.top = `${-y / 101 * 100}%`;
  crop.append(image);
  return crop;
}
function updateExplorer() {
  const {pretraining,model,reduction} = selection;
  const baseline = model === 'alex' || model === 'vgg';
  const ratioIndex = Math.max(0,ratios.indexOf(reduction));
  controls.forEach(button => {
    const active = String(selection[button.dataset.group]) === button.dataset.value;
    button.setAttribute('aria-pressed',String(active));
    if (button.dataset.group === 'reduction') button.disabled = baseline;
  });
  const modelName = {early:'early insertion',middle:'middle insertion',late:'late insertion',alex:'AlexNet',vgg:'VGG16'}[model];
  label.textContent = `${pretraining === 'face' ? 'Face' : 'Object'}-pretrained · ${modelName}${baseline ? '' : ' · R = ' + reduction}`;
  const x = groups[model] + (baseline ? 0 : ratioIndex * 101.4);
  const rowStart = pretraining === 'face' ? 76 : 541;
  const cards = conditions.map((condition,index) => {
    const card = document.createElement('article');
    card.className = 'compare-card';
    const title = document.createElement('h3');
    title.textContent = condition.name;
    const inputFigure = document.createElement('figure');
    const input = document.createElement('img');
    input.src = condition.input;
    input.alt = `Stimulus 11, ${condition.name.toLowerCase()}, without Grad-CAM overlay`;
    input.width = 224; input.height = 224; input.loading = 'lazy';
    const inputCaption = document.createElement('figcaption');
    inputCaption.textContent = 'Input example';
    inputFigure.append(input,inputCaption);
    const mapFigure = document.createElement('figure');
    mapFigure.append(makeHeatmap(x,rowStart+index*101.25,`Manuscript Grad-CAM example: ${label.textContent}, ${condition.name}`));
    const mapCaption = document.createElement('figcaption');
    mapCaption.textContent = 'Grad-CAM';
    mapFigure.append(mapCaption);
    card.append(title,inputFigure,mapFigure);
    return card;
  });
  grid.replaceChildren(...cards);
}
controls.forEach(button => button.addEventListener('click',() => {
  const group = button.dataset.group;
  selection[group] = group === 'reduction' ? Number(button.dataset.value) : button.dataset.value;
  updateExplorer();
}));
updateExplorer();
