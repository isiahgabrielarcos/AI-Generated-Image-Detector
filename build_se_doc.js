// Builds the Software Engineering Documentation .docx
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType,
  TableOfContents, PageBreak, PageNumber, Header, Footer
} = require("docx");

const CONTENT_W = 9360; // US Letter, 1" margins
const HDR = "1F3864", HDRTXT = "FFFFFF", ZEBRA = "EEF3FB", ACCENT = "2E75B6";
const border = { style: BorderStyle.SINGLE, size: 1, color: "B8C4D9" };
const borders = { top: border, bottom: border, left: border, right: border };

// ---------- helpers ----------
const T = (t, o = {}) => new TextRun({ text: t, ...o });
const P = (text, o = {}) =>
  new Paragraph({ spacing: { after: 120, line: 276 }, children: Array.isArray(text) ? text : [T(text)], ...o });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [T(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [T(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [T(t)] });
const bullet = (text) => new Paragraph({ numbering: { reference: "b", level: 0 }, spacing: { after: 60, line: 268 },
  children: Array.isArray(text) ? text : [T(text)] });
const num = (text) => new Paragraph({ numbering: { reference: "n", level: 0 }, spacing: { after: 60, line: 268 },
  children: Array.isArray(text) ? text : [T(text)] });
const spacer = () => new Paragraph({ spacing: { after: 60 }, children: [T("")] });

function cell(content, { w, head = false, fill, bold = false, align } = {}) {
  const kids = (Array.isArray(content) ? content : [content]).map((c) =>
    typeof c === "string"
      ? new Paragraph({ alignment: align, spacing: { after: 40, line: 264 },
          children: [T(c, { bold: head || bold, color: head ? HDRTXT : undefined, size: 19 })] })
      : c);
  return new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    shading: { fill: head ? HDR : (fill || "FFFFFF"), type: ShadingType.CLEAR },
    children: kids,
  });
}

// generic table: headers[], rows[][], colWidths[]
function table(headers, rows, colWidths) {
  const headRow = new TableRow({ tableHeader: true,
    children: headers.map((h, i) => cell(h, { w: colWidths[i], head: true })) });
  const bodyRows = rows.map((r, ri) =>
    new TableRow({ children: r.map((c, i) =>
      cell(c, { w: colWidths[i], fill: ri % 2 ? ZEBRA : "FFFFFF" })) }));
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: colWidths,
    rows: [headRow, ...bodyRows] });
}

// IPO 3-column table for a module
function ipo(inp, proc, out) {
  const w = [3120, 3120, 3120];
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: w,
    rows: [
      new TableRow({ tableHeader: true, children: [
        cell("Input", { w: w[0], head: true }), cell("Process", { w: w[1], head: true }),
        cell("Output", { w: w[2], head: true }) ] }),
      new TableRow({ children: [
        cell(inp, { w: w[0] }), cell(proc, { w: w[1] }), cell(out, { w: w[2] }) ] }),
    ] });
}

// module block: id+title heading, purpose, file, IPO, how/when
function moduleBlock(num, title, file, purpose, input, process, output, howwhen) {
  return [
    H2(`${num}  ${title}`),
    P([T("Source: ", { bold: true }), T(file, { font: "Consolas", size: 19 })]),
    P([T("Purpose. ", { bold: true }), T(purpose)]),
    ipo(input, process, output),
    spacer(),
    P([T("How and when to use. ", { bold: true }), T(howwhen)]),
  ];
}

const children = [];

// ===== TITLE PAGE =====
children.push(
  new Paragraph({ spacing: { before: 1600, after: 120 }, alignment: AlignmentType.CENTER,
    children: [T("Software Engineering Documentation", { bold: true, size: 48, color: HDR })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
    children: [T("Software Requirements, Architecture, Design, Metrics, and Testing", { size: 26, color: ACCENT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 },
    children: [T("Transformer Based on Multi-Domain Feature Fusion for AI-Generated Image Detection",
      { italics: true, size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [T("College of Computer and Information Sciences", { size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [T("Polytechnic University of the Philippines", { size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 },
    children: [T("BSCS 3-3  |  Group 4  |  COSC 304 - Introduction to AI", { size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
    children: [T("Arcos, Isiah Gabriel A.   |   Mendoza, Alron David V.   |   Morales, Drixelle L.", { size: 20 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
    children: [T("Nava, Denelle D.   |   Yabut, Natasha Julia S.", { size: 20 })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// ===== TOC =====
children.push(H1("Table of Contents"),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }));

// ===== 1. INTRODUCTION =====
children.push(H1("1. Introduction"));
children.push(P([T("Context. ", { bold: true }),
  T("This document specifies the software-engineering artifacts for the project, a faithful re-implementation and extension of the Man & Cho (2026) multi-domain feature-fusion framework for AI-generated image detection. The system performs binary classification of an input image as Real (human-made, label 0) or AI-generated (label 1) by combining two complementary feature domains, a frozen CLIP ViT-L/14 spatial-semantic branch and a Daubechies-4 (db4) wavelet frequency branch, fusing them through a Spatial-Frequency Cross-Domain Feature Fusion (SFDF) module, modelling global context with a Swin-Tiny Transformer, and classifying with a lightweight MLP head.")]));
children.push(P([T("This document covers four deliverables: (1) the Software Requirements Specification (SRS); (2) the Software Design; (3) the Software Architecture, stating every module with its Input-Process-Output (IPO) and usage guidance; and (4) Software Metrics and Testing, including module, integration, and functional test plans with example test cases.")]));

children.push(H2("1.1 Definitions, Acronyms, and Abbreviations"));
children.push(table(["Term", "Meaning"], [
  ["CLIP", "Contrastive Language-Image Pre-training; frozen ViT-L/14 used for spatial-semantic features"],
  ["ViT", "Vision Transformer (the CLIP image encoder backbone)"],
  ["DWT", "Discrete Wavelet Transform (Daubechies-4 / db4), used for frequency decomposition"],
  ["SFDF", "Spatial-Frequency Cross-Domain Feature Fusion (cross-attention alignment + gated integration)"],
  ["Swin", "Shifted-window Transformer backbone (Swin-Tiny configuration)"],
  ["Fs / Ff", "Spatial (CLIP) feature sequence / Frequency (wavelet) feature sequence"],
  ["ACC / AP / AUC", "Accuracy / Average Precision / Area Under ROC Curve (evaluation metrics)"],
  ["IPO", "Input-Process-Output module specification"],
  ["Grad-CAM", "Gradient-weighted Class Activation Map (explainability heat map)"],
], [2200, 7160]));

// ===== 2. SRS =====
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H1("2. Software Requirements Specification (SRS)"));

children.push(H2("2.1 Overall Description"));
children.push(P([T("Product perspective. ", { bold: true }),
  T("The system is a self-contained deep-learning pipeline with an optional deployment layer. It ingests image datasets, pre-computes frozen-CLIP features into a cache, trains the trainable components (wavelet CNN, SFDF, Swin backbone, MLP head) while the CLIP ViT remains frozen, evaluates per-generator generalization, and serves predictions with an explainability heat map through a local inference server intended to back a browser extension.")]));
children.push(P([T("User characteristics. ", { bold: true }),
  T("Primary users are the researchers who train and evaluate the model (technical), and end users who submit an image and receive a Real/AI-generated verdict with a confidence score and heat map (non-technical).")]));
children.push(P([T("Constraints and assumptions. ", { bold: true }),
  T("Training and inference run on CPU (no dedicated GPU is assumed); the frozen CLIP ViT-L/14 is the dominant memory/compute cost and is bypassed via a feature cache wherever possible. Input images are resized to 224 x 224 and normalized with CLIP statistics. Datasets must be organized into real/ and fake/ class folders.")]));

children.push(H2("2.2 Functional Requirements"));
children.push(table(["ID", "Requirement", "Priority"], [
  ["FR-01", "The system shall classify an input image as Real (0) or AI-generated (1) and output a probability in [0,1].", "High"],
  ["FR-02", "The system shall extract spatial-semantic features using a frozen CLIP ViT-L/14 backbone.", "High"],
  ["FR-03", "The system shall extract frequency-domain features using a 2-D db4 DWT and a lightweight CNN.", "High"],
  ["FR-04", "The system shall fuse the two feature streams via cross-attention alignment and gated integration (SFDF).", "High"],
  ["FR-05", "The system shall model global context with a Swin-Tiny Transformer and classify with an MLP head.", "High"],
  ["FR-06", "The system shall pre-compute and cache frozen-CLIP tokens to disk and reuse them during training/evaluation.", "High"],
  ["FR-07", "The system shall train using binary focal loss, AdamW with two learning-rate groups, warmup-cosine scheduling, gradient accumulation, gradient clipping, and early stopping.", "High"],
  ["FR-08", "The system shall report ACC, AP, Recall, F1, AUC, and a confusion matrix on a validation set.", "High"],
  ["FR-09", "The system shall evaluate detection performance independently per generator and produce Tables 1-3 (DFDC, GANs, diffusion).", "High"],
  ["FR-10", "The system shall support four ablation modes (Clip, Clip+F, Clip+F+A, Clip+F+A+G) from a single checkpoint.", "Medium"],
  ["FR-11", "The system shall generate a region-of-interest heat map (Grad-CAM) explaining each decision.", "Medium"],
  ["FR-12", "The system shall expose a local HTTP endpoint that returns a verdict, probability, and optional heat-map overlay.", "Medium"],
  ["FR-13", "The system shall verify dataset integrity (no train/test leakage) and equalize preprocessing across classes.", "Medium"],
], [900, 7460, 1000]));

children.push(H2("2.3 Non-Functional Requirements"));
children.push(table(["ID", "Category", "Requirement"], [
  ["NFR-01", "Performance", "A cached training epoch shall avoid running the CLIP ViT; a single heat map shall complete in about 1 second on CPU when tokens are cached."],
  ["NFR-02", "Correctness", "Cached CLIP features shall match a live forward (cosine similarity about 1.0) before being used for training."],
  ["NFR-03", "Reliability", "Training shall checkpoint the best model by validation AUC and support resuming from the last saved epoch."],
  ["NFR-04", "Efficiency", "Only the trainable components (about 34.7 M of 338.7 M parameters, 10.25%) shall receive gradient updates; the CLIP ViT shall stay frozen."],
  ["NFR-05", "Portability", "The system shall run on Windows/CPU without a GPU, using num_workers=0 to avoid subprocess overhead."],
  ["NFR-06", "Usability", "End users shall receive a clear Real/AI verdict, a confidence value, and a visual heat-map explanation."],
  ["NFR-07", "Maintainability", "Each architectural concern (spatial, frequency, fusion, backbone, head) shall be an independent, separately testable module."],
], [900, 1700, 6760]));

children.push(H2("2.4 External Interface Requirements"));
children.push(bullet([T("Data interface. ", { bold: true }), T("Dataset folders with real/ and fake/ subdirectories; CLIP cache files stored as .pt dictionaries {paths, features}.")]));
children.push(bullet([T("Configuration interface. ", { bold: true }), T("A YAML file (configs/default.yaml) supplies all model and training hyper-parameters.")]));
children.push(bullet([T("Network interface. ", { bold: true }), T("HTTP POST /detect accepting an image and returning JSON {prediction, probability, confidence, heatmap_overlay}; GET /health for status.")]));

// ===== 3. SOFTWARE DESIGN =====
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H1("3. Software Design"));

children.push(H2("3.1 Architectural Design"));
children.push(P("The system follows a layered, pipeline architecture. Each layer is composed of cohesive modules (specified in Section 4) and communicates through well-defined tensor or file contracts."));
children.push(table(["Layer", "Responsibility", "Modules"], [
  ["Data Layer", "Acquire, clean, equalize, and load images; guarantee test integrity.", "M1, M2, M3"],
  ["Feature Cache Layer", "Pre-compute and serve frozen-CLIP tokens to remove the ViT from the hot path.", "M4"],
  ["Model Layer", "Extract, fuse, model, and classify features (the detector network).", "M5-M10"],
  ["Training Layer", "Optimize trainable parameters and select the best model.", "M11, M12"],
  ["Evaluation Layer", "Quantify in-distribution and cross-generator performance.", "M13, M14"],
  ["Explainability & Deployment Layer", "Explain decisions and serve predictions.", "M15, M16"],
], [2300, 5060, 2000]));

children.push(H3("End-to-End Data Flow"));
children.push(P("Image [B,3,224,224] -> CLIP branch (Fs [B,256,768]) and Wavelet branch (Ff [B,256,768]) -> SFDF fusion (F_fused [B,256,768]) -> reshape to 2-D map [B,768,16,16] -> Swin backbone (z [B,768]) -> MLP head (logit [B,1]) -> sigmoid -> P(AI). The two branches must emit the same 256-token x 768-dim shape so the cross-attention can align them; this shape contract is the central integration constraint of the design."));

children.push(H2("3.2 Data Design"));
children.push(table(["Data Structure", "Format / Contract"], [
  ["Dataset folder", "<root>/real/*.png and <root>/fake/*.png (label 0 = real, 1 = fake)"],
  ["CLIP cache file", "{name}_clip.pt = dict{ paths: list[str], features: Tensor[N,256,1024] float16 }"],
  ["Per-generator cache", "pergen_{generator}_clip.pt, same schema, used only at evaluation"],
  ["Batch (cached)", "3-tuple (image [B,3,224,224], clip_tokens [B,256,1024], label [B])"],
  ["Batch (non-cached)", "2-tuple (image [B,3,224,224], label [B])"],
  ["Checkpoint", "dict{ epoch, model state_dict, optimizer state_dict, best_auc, metrics, cfg }"],
  ["Result artifacts", "results/.../tables.txt (formatted) and results.csv (per-generator ACC/AP/AUC)"],
], [2600, 6760]));

children.push(H2("3.3 Interface (Module) Design"));
children.push(P("Modules expose narrow interfaces. The detector accepts an optional clip_tokens argument so the same forward() serves cached training, cached evaluation, and live inference. build_detector() auto-detects the cache and sets load_visual=False to skip loading the 1.2 GB ViT. A single ablation flag selects the fusion path, so all four model variants are produced from one trained checkpoint."));

// ===== 4. SOFTWARE ARCHITECTURE - MODULES =====
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H1("4. Software Architecture: Module Specifications"));
children.push(P("This section states every module of the system and discusses it fully. Each module is given its source, purpose, an Input-Process-Output (IPO) table, and guidance on how and when to use it."));

children.push(H2("4.0 Module Map"));
children.push(table(["ID", "Module", "Layer"], [
  ["M1", "Data Equalization Module", "Data"],
  ["M2", "Data Leakage Verification Module", "Data"],
  ["M3", "Dataset & DataLoader Module", "Data"],
  ["M4", "CLIP Feature Cache Module", "Feature Cache"],
  ["M5", "Spatial-Semantic Feature Extraction Module (CLIP)", "Model"],
  ["M6", "Frequency-Domain Feature Extraction Module (Wavelet)", "Model"],
  ["M7", "Cross-Domain Feature Fusion Module (SFDF)", "Model"],
  ["M8", "Global Contextual Backbone Module (Swin)", "Model"],
  ["M9", "Classification Head Module (MLP)", "Model"],
  ["M10", "Detector Orchestration Module", "Model"],
  ["M11", "Loss Module (Binary Focal Loss)", "Training"],
  ["M12", "Training Module", "Training"],
  ["M13", "Evaluation Metrics Module", "Evaluation"],
  ["M14", "Per-Generator Evaluation Module", "Evaluation"],
  ["M15", "Explainability / Heat-Map Module", "Explainability"],
  ["M16", "Inference & Deployment Server Module", "Deployment"],
], [800, 6560, 2000]));

// Module blocks
const M = [];
M.push(...moduleBlock("M1", "Data Equalization Module", "equalize_training_data.py",
  "Removes the real-vs-fake 'source shortcut' by passing every training image (both classes) through one identical preprocessing pipeline, so the only feature distinguishing real from fake is the generation artifact, not resolution or file format. This is an extension beyond Man & Cho added after diagnosing that raw datasets leaked class identity through resolution/format.",
  "Raw dataset folders (real/, fake/) of mixed resolutions and formats; target size (224) and format (PNG).",
  "Resize-shortest-side then center-crop to 224 x 224; re-encode both classes to a single lossless PNG format; write a cleaned copy non-destructively to datasets_eq/.",
  "Equalized dataset (datasets_eq/<set>/real, fake) with uniform 224 x 224 PNG images; originals untouched.",
  "Run once, before building the CLIP cache, whenever new raw data is added. Use it when real and fake images differ in resolution/format, which would otherwise let the model cheat."));
M.push(...moduleBlock("M2", "Data Leakage Verification Module", "check_leakage.py",
  "Guarantees test-set integrity by detecting any image that appears in both the training set and the per-generator evaluation set, using exact (MD5) and perceptual (pHash) hashing.",
  "Training root and per-generator evaluation root.",
  "Hash all training images; hash all evaluation images; report exact duplicates and near-duplicates with their source paths.",
  "A leakage report and a count of duplicates (target: 0).",
  "Run after dataset construction and before reporting any results. Use it to certify that cross-generator scores are not inflated by leaked images."));
M.push(...moduleBlock("M3", "Dataset & DataLoader Module", "data/dataset.py",
  "Loads images and labels, applies the preprocessing/augmentation transforms, attaches the CLIP cache when present, and batches data for training and evaluation.",
  "Image folders; configuration (split ratio, augmentation flags); optional ClipFeatureCache.",
  "Build train/eval transforms (augment only for training: random flip, resized crop, Gaussian blur, JPEG compression); label by folder; serve 2-tuples or 3-tuples; collate via clip_collate_fn; split 80/20.",
  "PyTorch DataLoaders yielding (image, [clip_tokens], label) batches.",
  "Used by the Training and Evaluation layers. Provide a cache to enable the fast 3-tuple path; omit it to compute CLIP live. Augmentation is active only on the training split."));
M.push(...moduleBlock("M4", "CLIP Feature Cache Module", "cache_clip_features.py + ClipFeatureCache",
  "Pre-computes the frozen CLIP patch tokens for every image once and stores them on disk, then serves them by image path at train/eval time. Because the ViT is frozen, its output is deterministic, so caching removes the single largest CPU cost.",
  "Equalized image folders; the frozen CLIP extractor; output cache directory.",
  "Run the ViT once per image (no gradient), store {paths, features [N,256,1024] float16}; at runtime, load into RAM and look up tokens by path. A verification step confirms cached features match a live forward (cosine about 1.0).",
  "Cache files {name}_clip.pt and pergen_{generator}_clip.pt; in-RAM path-to-token lookup.",
  "Build once before training (and once for per-generator evaluation). Always verify a sample (cosine about 1.0) before trusting a freshly built cache, to catch any extraction bug early."));
M.push(...moduleBlock("M5", "Spatial-Semantic Feature Extraction Module (CLIP)", "models/clip_extractor.py",
  "Extracts high-level semantic features from the image using a frozen CLIP ViT-L/14 (QuickGELU, OpenAI weights). Only a small trainable linear projection (1024 -> 768) adapts CLIP space to the shared dimension; the ViT itself is never updated.",
  "Image [B,3,224,224], or pre-computed clip_tokens [B,256,1024].",
  "If tokens are supplied, skip the ViT; otherwise run conv patch embedding, prepend CLS, add positional embedding, LayerNorm, batch-first transformer, drop CLS. Project tokens 1024 -> 768.",
  "Spatial feature sequence Fs [B,256,768].",
  "Used as the spatial branch of the detector. Supply cached tokens for speed; load the ViT (load_visual=True) only for live inference on new images such as in the deployment server."));
M.push(...moduleBlock("M6", "Frequency-Domain Feature Extraction Module (Wavelet)", "models/wavelet_extractor.py",
  "Captures subtle, low-level frequency artifacts that semantic methods miss. A fixed 2-D db4 DWT decomposes the image; a small trainable CNN encodes the high-frequency sub-bands into a token sequence matching Fs.",
  "Image [B,3,224,224].",
  "Per channel (R,G,B), apply db4 DWT, discard the LL sub-band, keep LH/HL/HH (9-channel map at H/2 x W/2); encode with three Conv-BatchNorm-ReLU stride-2 blocks (9->64->256->768); adaptive-average-pool to a 16 x 16 grid; flatten.",
  "Frequency feature sequence Ff [B,256,768].",
  "Used as the frequency branch of the detector. The DWT is a fixed buffer (not trained); the CNN is trained at the higher learning rate (1e-4) and receives the augmented image so it learns artifacts robust to compression/blur/resize."));
M.push(...moduleBlock("M7", "Cross-Domain Feature Fusion Module (SFDF)", "models/sfdf.py",
  "Fuses the spatial and frequency streams in two stages: cross-attention alignment (spatial as query, frequency as key/value) followed by gated integration that adaptively balances the two domains per token.",
  "Fs [B,256,768] and Ff [B,256,768]; a use_gate flag.",
  "Stage 1: multi-head cross-attention A = softmax(QK^T/sqrt(d))V with Q from Fs and K,V from Ff. Stage 2: gate G = sigmoid(MLP([Fs;A])); F_fused = G*Fs + (1-G)*A. If use_gate is false, F_fused = Fs + A.",
  "Fused feature sequence F_fused [B,256,768].",
  "Used after both branches. Enable gating for the full model; disable it for the Clip+F+A ablation. The module is the heart of the 'multi-domain fusion' contribution."));
M.push(...moduleBlock("M8", "Global Contextual Backbone Module (Swin)", "models/swin_backbone.py",
  "Models global context and long-range dependencies over the fused feature map using a Swin-Tiny Transformer with window and shifted-window attention.",
  "Fused 2-D feature map [B,768,16,16].",
  "Project channels to the Swin embedding dim (1x1 conv); process through four Swin stages (depths [2,2,6,2], heads [3,6,12,24], window 4) using W-MSA/SW-MSA, LayerNorm, MLP, residuals; global-average-pool.",
  "Pooled global representation z [B,768].",
  "Used after fusion as the main learning backbone. Window size 4 evenly divides the 16 x 16 grid; patch_size 1 preserves the token resolution since the input is already a token grid, not raw pixels."));
M.push(...moduleBlock("M9", "Classification Head Module (MLP)", "models/detector.py (classifier)",
  "Maps the pooled representation to a single logit for binary classification.",
  "Pooled vector z [B,768].",
  "Two-layer MLP 768 -> 384 (GELU, Dropout 0.3) -> 1; sigmoid converts the logit to P(AI-generated).",
  "Logit [B,1]; probability after sigmoid.",
  "Used as the final stage of every forward pass and ablation mode. Dropout provides regularization against in-distribution overfitting."));
M.push(...moduleBlock("M10", "Detector Orchestration Module", "models/detector.py (AIGCDetector, build_detector)",
  "Wires all model modules into one network, exposes the ablation switch, and constructs the model from configuration with cache-aware loading.",
  "Configuration dict; optional ablation mode; optional force_load_visual flag.",
  "Instantiate CLIP, Wavelet, SFDF, Swin, and head; in forward(), route through the fusion path selected by the ablation flag; reshape the token sequence to a 2-D map for Swin; auto-detect the cache to decide whether to load the ViT.",
  "An AIGCDetector producing logits, optionally returning F_fused for the heat map.",
  "The single entry point for building and running the model. Use force_load_visual=True for live inference (server, heat map on new images); leave it false to use cached tokens during training/evaluation."));
M.push(...moduleBlock("M11", "Loss Module (Binary Focal Loss)", "losses/focal_loss.py",
  "Computes the training objective, down-weighting easy examples so the model focuses on hard, near-boundary images.",
  "Logits [B,1] and float labels [B] in {0,1}; focusing parameter gamma (=2).",
  "Compute p = sigmoid(logit), p_t, stable BCE, then multiply by the focal weight (1 - p_t)^gamma; average.",
  "Scalar loss value.",
  "Used inside the training loop. Increase gamma to emphasize hard examples more; gamma = 2 matches the paper."));
M.push(...moduleBlock("M12", "Training Module", "train.py",
  "Drives the full training process: data loading, optimization, scheduling, checkpointing, validation, and early stopping.",
  "Configuration; optional resume checkpoint.",
  "Build dataloaders and model; AdamW with two LR groups (wavelet CNN 1e-4, Swin+projection 1e-5); linear warmup then cosine decay; gradient accumulation (effective batch 32) and gradient clipping (1.0); validate each epoch; save best by AUC; early-stop on AUC plateau (patience 15).",
  "Trained checkpoints (best_model.pt, periodic epoch_XXX.pt) and TensorBoard logs.",
  "Run to train or resume the model. Select the final model by cross-generator score, not only validation AUC, because in-distribution AUC can keep rising while cross-generator performance plateaus."));
M.push(...moduleBlock("M13", "Evaluation Metrics Module", "utils/metrics.py",
  "Computes and reports the standard detection metrics from ground-truth labels and predicted probabilities.",
  "y_true (0/1) and y_prob ([0,1]); decision threshold (0.5).",
  "Compute Accuracy, Average Precision, Recall, F1, AUC-ROC, and the confusion matrix; handle the single-class edge case for AUC.",
  "A metrics dictionary plus pretty-printed confusion matrix.",
  "Used by the training loop (per epoch) and by per-generator evaluation. AP is the headline metric alongside ACC; AUC is used for checkpoint selection."));
M.push(...moduleBlock("M14", "Per-Generator Evaluation Module", "evaluate_per_generator.py",
  "Measures cross-generator generalization by evaluating each generator's folder independently and producing the paper's Tables 1-3 (DFDC, GAN-based, diffusion-based).",
  "A checkpoint; the per-generator dataset root; the per-generator CLIP cache; optional ablation mode.",
  "For each generator, load its cache, run inference, compute ACC/AP/AUC; group results into the three tables with per-table means; write formatted tables and CSV.",
  "results/.../tables.txt and results.csv with per-generator and mean ACC/AP.",
  "Run after training to report generalization, and as a fast sanity check during training (caches make it about 10 minutes). Use --ablation to produce all four variants from one checkpoint."));
M.push(...moduleBlock("M15", "Explainability / Heat-Map Module", "utils/visualization.py",
  "Produces a region-of-interest heat map (Man & Cho Figure 6 style) explaining which regions drive the Real/AI decision, with an efficient design that keeps the frozen ViT out of the autograd graph.",
  "A trained model; an image (or cached tokens); a method (gradcam or fast).",
  "Grad-CAM: gradient-weighted activations of the fused feature map, ReLU, upsample; or a forward-only activation-magnitude map. Extract CLIP tokens once under no_grad (or reuse cache) so backprop traverses only the 34.7 M trainable head.",
  "A [224,224] heat map in [0,1] and a colour overlay on the original image.",
  "Used by the deployment server and for paper figures. Use 'gradcam' for faithful saliency; use 'fast' (or cached tokens) for interactive/real-time use (about 0.16-0.8 s on CPU)."));
M.push(...moduleBlock("M16", "Inference & Deployment Server Module", "server.py",
  "Serves predictions over HTTP so a browser extension or web client can request real-time detection with an explanation.",
  "HTTP POST /detect with an image and a generate_heatmap flag.",
  "Decode and preprocess the image; build the detector with the ViT loaded (live inference on arbitrary images); run the forward pass; optionally generate a heat-map overlay; return JSON.",
  "JSON {prediction, probability, confidence, processing_time_ms, heatmap_overlay}.",
  "Run to deploy the trained model. It loads the ViT (force_load_visual=True) because browser images have no pre-built cache; pass cached tokens only for offline batch use."));
M.forEach((c) => children.push(c));

// ===== 5. SOFTWARE METRICS =====
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H1("5. Software Metrics"));
children.push(P("The following metrics quantify the size, capacity, and runtime characteristics of the system. They are used to assess complexity, computational cost, and deployability."));

children.push(H2("5.1 Model Capacity Metrics (Parameter Summary)"));
children.push(table(["Component", "Total Params", "Trainable", "% of Total"], [
  ["Spatial - CLIP Backbone (frozen)", "303,966,208", "0", "89.75%"],
  ["Spatial - Projection Layer", "786,432", "786,432", "0.23%"],
  ["Frequency - DWT (db4, fixed)", "0", "0", "0.00%"],
  ["Frequency - CNN Encoder", "1,924,288", "1,924,288", "0.57%"],
  ["Fusion - SFDF Module", "4,133,376", "4,133,376", "1.22%"],
  ["Backbone - Swin-Tiny", "27,579,402", "27,579,402", "8.14%"],
  ["Head - MLP Classifier", "295,681", "295,681", "0.09%"],
  ["TOTAL", "338,685,387", "34,719,179", "100.00%"],
], [3360, 2200, 2000, 1800]));
children.push(P([T("Interpretation. ", { bold: true }), T("Only 10.25% of parameters are trainable (34.7 M of 338.7 M); the remaining ~304 M are the frozen CLIP ViT. This is the source of the model's data efficiency: it adapts CLIP's prior knowledge rather than re-learning it. The Swin-Tiny backbone dominates the trainable budget (about 79% of trainable parameters).")]));

children.push(H2("5.2 Efficiency and Runtime Metrics"));
children.push(table(["Metric", "Definition", "Notes / Target"], [
  ["Latency (ms/batch)", "Time from input submission to returned prediction.", "Lower is better; report on CPU."],
  ["Throughput (img/s)", "Images classified per second.", "Higher is better; measured in batches."],
  ["Inference Memory (MB)", "Peak memory while classifying.", "ViT skipped via cache saves ~1.2 GB."],
  ["Heat-map time (s)", "Time to produce one Grad-CAM/overlay.", "~0.8 s (Grad-CAM) / ~0.16 s (fast) with cached tokens."],
  ["Cache build time", "One-time cost to pre-compute CLIP tokens.", "Amortized across all epochs."],
  ["Trainable params (M)", "Learnable values updated during training.", "34.7 M (lightweight relative to total)."],
], [2200, 4160, 3000]));
children.push(P([T("Note. ", { bold: true }), T("Latency, throughput, and memory must be populated with actual measurements once final training completes; the table above defines the measurement protocol.")]));

// ===== 6. TESTING =====
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H1("6. Software Metrics and Testing"));
children.push(H2("6.1 Testing Strategy and Objectives"));
children.push(P("Testing proceeds bottom-up in three levels. Module (unit) testing validates each module in isolation against its IPO contract. Integration testing validates that connected modules interoperate correctly across their shared contracts (shapes, caches, checkpoints). Functional (system) testing validates end-to-end behaviour against the functional requirements. The objective is to confirm correctness, the integrity of the data pipeline, and that cross-generator generalization is genuine."));
children.push(table(["Level", "Goal", "Example Test Cases", "Pass Criterion"], [
  ["Module / Unit", "Each module satisfies its IPO contract in isolation.", "42", "Output shape/type/values match the specification."],
  ["Integration", "Connected modules interoperate across shared contracts.", "14", "Data flows end-to-end with no shape/cache/label mismatch."],
  ["Functional / System", "End-to-end behaviour meets the functional requirements.", "16", "User-visible outcome matches the requirement."],
  ["TOTAL", "Full coverage across the pipeline.", "72", "All planned cases executed and passed."],
], [2100, 3360, 1900, 2000]));

children.push(H2("6.2 Module (Unit) Testing Plan"));
children.push(P("42 unit test cases are planned across the 16 modules (about 2-4 per module). Representative cases:"));
children.push(table(["Test ID", "Module", "Input", "Expected Output"], [
  ["UT-01", "M6 DWT", "Image [1,3,224,224]", "High-freq map [1,9,112,112] (LL discarded, 9 channels)"],
  ["UT-02", "M6 Wavelet", "Image [1,3,224,224]", "Ff sequence [1,256,768]"],
  ["UT-05", "M5 CLIP", "Model with load_visual=True", "All ViT parameters have requires_grad = False"],
  ["UT-06", "M5/M4 CLIP", "Cached feature vs live forward", "Cosine similarity about 1.0 (correctness gate)"],
  ["UT-09", "M7 SFDF", "Fs, Ff [1,256,768], use_gate=True", "F_fused [1,256,768]; gate values within (0,1)"],
  ["UT-10", "M7 SFDF", "use_gate=False", "F_fused equals Fs + A"],
  ["UT-13", "M8 Swin", "Fused map [1,768,16,16]", "Pooled vector [1,768]"],
  ["UT-15", "M9 Head", "z [1,768]", "Logit [1,1]"],
  ["UT-18", "M11 Loss", "Confident-correct logits", "Loss near 0; non-negative for all inputs"],
  ["UT-22", "M13 Metrics", "Perfect predictions", "ACC = 1.0, AUC = 1.0; single-class input -> AUC NaN handled"],
  ["UT-27", "M3 Transforms", "Eval transform applied twice", "Identical output (deterministic, no augmentation)"],
  ["UT-31", "M1 Equalize", "Mixed-size real & fake images", "All outputs 224x224 PNG for both classes"],
  ["UT-34", "M2 Leakage", "Dataset with one injected duplicate", "Duplicate detected and reported"],
  ["UT-38", "M15 Heat map", "Trained model + one image", "Heat map [224,224] with values in [0,1]"],
], [1100, 1500, 3260, 3500]));

children.push(H2("6.3 Integration Testing Plan"));
children.push(P("14 integration test cases verify the contracts between connected modules. Representative cases:"));
children.push(table(["Test ID", "Modules", "Scenario", "Expected Output"], [
  ["IT-01", "M4 <-> M3", "Build cache, then look up tokens by path in the DataLoader", "Path match rate 100% (e.g., 2000/2000); served token shape [256,1024]"],
  ["IT-03", "M4 <-> M10", "Detector built with load_visual=False, fed cached tokens", "forward() runs without loading the ViT and returns a logit"],
  ["IT-04", "M5+M6 <-> M7 <-> M8", "Two branches -> SFDF -> Swin", "Shapes align end-to-end; pooled vector [B,768]"],
  ["IT-06", "M12 <-> M11", "One accumulation cycle of train_one_epoch on a tiny batch", "Loss decreases; optimizer steps after accum_steps"],
  ["IT-07", "M12 optimizer", "Inspect parameter groups", "Wavelet CNN at 1e-4; Swin+projection at 1e-5; ViT absent"],
  ["IT-09", "M12 checkpoint", "Save then resume", "Epoch, optimizer state, and best_auc restored exactly"],
  ["IT-11", "M14 <-> M4", "Per-generator eval reads pergen cache via GeneratorDataset", "All generators load; per-table ACC/AP computed"],
  ["IT-12", "M10 ablation", "Evaluate all four modes from one checkpoint", "Four valid result sets (Clip, Clip+F, Clip+F+A, full)"],
  ["IT-14", "M16 <-> M10/M15", "POST /detect with generate_heatmap=true", "JSON with probability and a heat-map overlay"],
], [1100, 1900, 3360, 3000]));

children.push(H2("6.4 Functional (System) Testing Plan"));
children.push(P("16 functional test cases validate end-to-end behaviour against the functional requirements. Representative cases:"));
children.push(table(["Test ID", "Requirement", "Scenario", "Expected Outcome"], [
  ["FT-01", "FR-01", "Submit a known real and a known AI image", "Correct Real/AI verdict with probability in [0,1]"],
  ["FT-02", "FR-07/08", "Train for several epochs", "Validation AUC increases; best_model.pt saved"],
  ["FT-03", "FR-09", "Run per-generator evaluation", "Tables 1-3 and CSV produced; GAN generators score well above chance"],
  ["FT-05", "NFR-01", "Robustness test: JPEG, blur, resize the test set at graded levels", "Accuracy degrades gracefully, not collapsing"],
  ["FT-07", "FR-11", "Generate heat maps for a real and an AI image", "Overlays highlight decision-relevant regions"],
  ["FT-09", "NFR-02", "Verify a freshly built cache", "Cosine about 1.0 vs live; otherwise training is blocked"],
  ["FT-11", "FR-13", "Run leakage check on final datasets", "Zero train/test duplicates"],
  ["FT-13", "FR-12", "Call /health and /detect on the server", "Healthy status; valid prediction JSON within latency target"],
  ["FT-15", "NFR-03", "Interrupt and resume training", "Training continues from the last epoch with no loss of state"],
], [1100, 1500, 3460, 3300]));

children.push(H2("6.5 Test Summary"));
children.push(P("In total, 72 test cases are planned: 42 module/unit, 14 integration, and 16 functional/system. Each level must be fully executed and pass before proceeding to the next. The correctness gate (cached features matching a live forward, cosine about 1.0) and the leakage gate (zero duplicates) are mandatory pre-conditions for any reported result."));

// ---------- document ----------
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, color: HDR, font: "Arial" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, color: ACCENT, font: "Arial" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, color: "404040", font: "Arial" },
        paragraph: { spacing: { before: 140, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 260 } } } }] },
      { reference: "n", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 260 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [ new Paragraph({ alignment: AlignmentType.CENTER,
      children: [ T("Software Engineering Documentation  |  Group 4  |  Page ", { size: 16, color: "808080" }),
        new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "808080" }) ] }) ] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("Software_Engineering_Documentation.docx", buf);
  console.log("WROTE Software_Engineering_Documentation.docx", buf.length, "bytes");
});
