# Vision Tasks in Computer Vision

## Core Computer Vision Tasks

### 1. Image Classification
Classification assigns a single label to an entire image, answering the question: *"What is in this image?"*

- **Input**: An image
- **Output**: A class label (e.g., "forest", "urban area", "flooded road")
- **Approach**: A CNN or Vision Transformer encodes the whole image into a feature vector, which is passed to a classifier head.
- **Common metrics**: Top-1 / Top-5 accuracy

### 2. Object Detection
Detection identifies **where** objects are and **what** they are, answering: *"What objects are in the image and where?"*

- **Input**: An image
- **Output**: A set of bounding boxes + class labels + confidence scores
- **Approach**: Two-stage models (Faster R-CNN) first propose regions, then classify them. One-stage models (YOLO, DETR) predict boxes and classes in a single pass.
- **Common metrics**: mAP (mean Average Precision) at IoU thresholds

### 3. Semantic Segmentation
Segmentation assigns a class label to **every pixel** in the image, answering: *"What class does each pixel belong to?"*

- **Input**: An image
- **Output**: A pixel-wise label map (same H×W as input)
- **Approach**: Encoder-decoder architectures (U-Net, DeepLab, SegFormer) produce dense predictions. Each pixel gets a class.
- **Common metrics**: mIoU (mean Intersection over Union), pixel accuracy

### Key Differences

| Task | Granularity | Output | Typical Use |
|---|---|---|---|
| Classification | Image-level | Single label | Scene type |
| Detection | Object-level | Boxes + labels | Locate objects |
| Segmentation | Pixel-level | Dense label map | Delineate boundaries |

---

## Application to Natural Disaster Scenarios

In the context of natural disasters analyzed from remote sensing imagery, each task serves a distinct operational role:

### Classification in Disaster Contexts
Disaster-scene classifiers can determine the **type of disaster** (flood, wildfire, earthquake, hurricane) or the **damage severity** (none, minor, major, destroyed) from a satellite image patch. The DisasterM3 benchmark includes disaster type recognition (DTR) and disaster scene recognition (DSR) as classification-style tasks.

### Detection in Disaster Contexts
Object detection is used to **locate affected structures** — damaged buildings, blocked roads, displaced vehicles. Bounding-box outputs allow rescue teams to spatially index where intervention is most needed. Detection also feeds into counting tasks (how many damaged buildings in a given zone?).

### Segmentation in Disaster Contexts
Pixel-level segmentation provides the most detailed analysis:
- **Building damage masks** at instance level (intact / damaged / destroyed), as used in DisasterM3 and xBD
- **Road accessibility maps** (intact / flooded / debris-covered)
- **Referring segmentation** — segmenting the specific object described in a natural-language query (e.g., "segment the buildings within 100m of lava flow")
- **Flood extent mapping** from SAR or optical imagery

Segmentation output is also used as a **visual prompt** in VQA pipelines (e.g., EarthVQA's SOBA framework), where semantic masks guide question answering by providing spatial context that global features alone would miss.

---

*This document covers classification, detection, and segmentation as the three foundational tasks. Advanced derivatives (instance segmentation, panoptic segmentation, change detection) combine or extend these primitives and are commonly encountered in disaster-response benchmarks.*
