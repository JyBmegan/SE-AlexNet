# SE-AlexNet Research Site

This folder contains a static project page for the SE-AlexNet study. Open `index.html` in a browser to view it locally, or publish the whole `research-site` folder with GitHub Pages or another static host.

The page is structured as a research project page rather than a portfolio: visual claim, mechanism, ROI evidence, grouped results, citation, and access instructions.

Primary links on the page:

- Paper: `assets/paper/manuscript-anonymous.pdf`
- Code page repository: https://github.com/JyBmegan/se-alexnet-research-site
- Model weights: request access only while the manuscript is under review

The page uses real project outputs from:

- `SE-AlexNet/0_InputImages`
- `SE-AlexNet/2_Results`
- `SE-AlexNet/5_FittingCurveAnalysis/Results`
- `GradCAM-ROI-SaliencyMapDecoder/Results`

## Citation

Use the review-stage manuscript citation for this work until the paper has a DOI, preprint, or journal page. The DFEW dataset citation is included on the webpage and links back to the official DFEW cite page.

## Model weights

Suggested storage:

- Put model weights in an encrypted archive before uploading to Google Drive.
- Share the Drive file only with approved institutional email addresses.
- Send the archive password in a separate email after approving the requester.

Example on macOS:

```bash
zip -er SE-AlexNet-weights.zip weights/
```

Suggested public wording:

> Model weights are available from the corresponding author upon reasonable request and subject to review-stage access restrictions.

Suggested request email:

```text
Subject: Request access to SE-AlexNet model weights

Name: <your first and last name>
Affiliation: <university or institute>
Department: <department>
Position: <job title>
Institutional email: <university email>

I request access to the SE-AlexNet model weights for research purposes. I will not reproduce, redistribute, sell, or make the weights available to any third party.
```
