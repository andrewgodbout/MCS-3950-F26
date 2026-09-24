# MCS 3950 — Computer Vision

Course materials for **MCS 3950**, University of Prince Edward Island, Fall 2026.

Instructor: Andrew Godbout · `agodbout@upei.ca` · CSH 406
Lectures: MWF 11:30–12:20, HSC 105

---

## Lecture Notebooks

Worked examples that accompany the Monday/Wednesday lectures. Run them alongside
the slides, or afterwards as a reference.

<!-- LECTURES:START -->

| Lecture | Date | Topic | |
|---|---|---|---|
| **L2** | Mon Sept 14 | Digital images — pixel representation, colour models (RGB, HSV, CIE Lab), bit depth, brightness arithmetic | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrewgodbout/MCS-3950-F26/blob/main/Notebooks/L02-Digital-Images/L2_Digital_Images.ipynb) |
| **L3** | Wed Sept 16 | Spatial filtering — convolution, box/Gaussian/bilateral filters, sharpening, border handling | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrewgodbout/MCS-3950-F26/blob/main/Notebooks/L03-Spatial_Filtering/L3_Spatial_Filtering.ipynb) |
| **L4** | Mon Sept 21 | Edge detection — image gradients, Sobel & Prewitt, Laplacian of Gaussian, Canny, non-maximum suppression | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrewgodbout/MCS-3950-F26/blob/main/Notebooks/L04-EdgeDetection/L4_Edge_Detection.ipynb) |
| **L5** | Wed Sept 23 | Image segmentation — global/adaptive thresholding, Otsu's method, region growing, morphology, watershed | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrewgodbout/MCS-3950-F26/blob/main/Notebooks/L05-Segmentation/L5_Segmentation.ipynb) |

<!-- LECTURES:END -->

---

## Active-Learning Labs

Friday sessions are hands-on labs. Click **Open in Colab** to run a lab in your
browser — no local install required. Labs are not submitted or graded; they build
the skills assessed in the tests, assignments, and project.

<!-- LABS:START -->

| Lab | Date | Topic | |
|---|---|---|---|
| **AL1** | Fri Sept 11 | Image processing fundamentals — arrays, colour spaces, ROI extraction, vegetation indices | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrewgodbout/MCS-3950-F26/blob/main/Notebooks/AL01-Introduction/AL1_Image_Processing_Fundamentals.ipynb) |
| **AL2** | Fri Sept 18 | Convolution and filtering — kernel design, denoising benchmark, preprocessing real PEI aerial imagery | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrewgodbout/MCS-3950-F26/blob/main/Notebooks/AL02-Filtering/AL2_Convolution_Filtering1.ipynb) |
| **AL3** | Fri Sept 25 | Morphology & watershed — structuring elements; distance-transform seeds to count touching objects; audit field sizes on 1935 imagery | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrewgodbout/MCS-3950-F26/blob/main/Notebooks/AL03-Segmentation/AL03-Seg.ipynb) |

<!-- LABS:END -->

---

## Working in Colab

1. Click the **Open in Colab** badge above.
2. Run the cells in order with `Shift + Enter`.
3. To keep your work: **File → Save a copy in Drive**. Edits made directly in the
   opened notebook are *not* saved back here.

Images used by the labs are served from `Notebooks/data/` in this repository, so
they stay available for the whole semester. Notebooks download what they need on
first run.

If a lab runs out of class time, finish it before the next session and bring
questions to the start of that class.

---

## Repository layout

```
Notebooks/
  AL01-Introduction/   lab notebooks
  data/                images and archives the notebooks download
```

---

Course materials © 2026 Andrew Godbout. Aerial photography of Prince Edward
Island is used for educational purposes; imagery sources are credited in the
notebooks where they appear.
