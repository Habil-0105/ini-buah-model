# Project and dataset context

This handoff describes the files present at the project root on 2026-10-20. The root contains five fruit dataset collections. It is not a Git checkout (`git status` reports that this directory is not a repository), and no root-level README or general project configuration was found.

## Structure and inventory

```text
.
├── Banana/
│   ├── Banana Classification Dataset/{Ambon,Cavendish,Mas,Raja,Saba}/
│   └── Banana Ripeness Classification Dataset-2/{Half_Ripe,Overripe,Ripe,Unripe}/
├── DragonFruit/
│   ├── Maturity_Grading/{Immature Dragon Fruit,Mature Dragon Fruit}/
│   └── Quality_Grading/{Defect Dragon Fruit,Fresh Dragon Fruit}/
├── Mango Shelf-Life Dataset/{Fully Ripe,Overripe,Perished,Semiripe,Unripe}/
├── avocado_hass_by_class/
│   ├── {Training,Testing}/{1_unripe,2_semiripe,3_ripe,4_overripe,5_rotten}/
│   ├── build_by_class.py, labels_matched.csv, unmatched.csv, summary.txt
└── papaya/
    ├── {train,test}/{images,labels}/
    ├── papaya_by_class/{Training,Testing}/{1_unripe,2_semiripe,3_ripe,4_rotten}/
    ├── papaya_by_class/{papaya_crops.py,crops.csv,summary.txt}
    ├── {README.dataset.txt,README.roboflow.txt,data.yaml}
    └── (macOS .DS_Store files)
```

Folder purpose is directly indicated by the names and contents: Banana has a cultivar-organized collection and a second ripeness-organized collection; DragonFruit separates maturity grading from quality grading; mango is organized by shelf-life/ripeness labels; avocado and papaya have class-organized image folders plus tabular summaries/manifests. No separate source-code or model-training project is present beyond the two conversion scripts noted below.

Inventory: image collections are JPEG (`.jpg`). Papaya also has YOLO text annotation files (`.txt`), two README-style text documents, and a YAML dataset configuration. Avocado has one Python script, two CSVs and a summary text file. Papaya has one Python crop script, one CSV and a summary text file. There are macOS `.DS_Store` files. No notebooks, model weights, or image formats other than JPEG were found in the inspected tree.

| Collection | Present JPEGs | Class/folder organization |
|---|---:|---|
| Banana / Banana Classification Dataset | 1,960 | `Ambon` 468, `Cavendish` 380, `Mas` 376, `Raja` 424, `Saba` 312 |
| Banana / Banana Ripeness Classification Dataset-2 | 1,960 | `Half_Ripe` 488, `Overripe` 448, `Ripe` 616, `Unripe` 408 |
| DragonFruit / Maturity_Grading | 2,127 | `Immature Dragon Fruit` 1,241; `Mature Dragon Fruit` 886 |
| DragonFruit / Quality_Grading | 1,652 | `Defect Dragon Fruit` 754; `Fresh Dragon Fruit` 898 |
| Mango Shelf-Life Dataset | 4,428 | `Fully Ripe` 841, `Overripe` 896, `Perished` 184, `Semiripe` 1,157, `Unripe` 1,350 |
| avocado_hass_by_class | 14,710 | `Training` 11,752; `Testing` 2,958 across five classes |
| papaya | 3,902 | raw detection export 1,951 (train 1,751/test 200); cropped class set 1,951 |

Counts exclude `.DS_Store` and non-image files. Papaya train/test counts refer to images; each has a matching number of label text files in its respective folder.

## Naming and dataset representation

- **Banana:** one collection uses cultivar folders and filenames such as `Ambon_Ripe_Left_0014.jpg`, `Cavendish_Half_Ripe_Right_0019.jpg`, `Mas_Half_Ripe_Bottom_0020.jpg`. Names visibly combine cultivar, ripeness term, view/side term, and a numeric suffix separated by underscores. Case is inconsistent in view terms (e.g. `Left` and `left`). The second collection sorts the same filename family into ripeness folders (`Half_Ripe`, `Overripe`, `Ripe`, `Unripe`).
- **DragonFruit:** class folders provide the labels. Example names are `Immature_Dragon_Original_Data0564.jpg`, `Mature_Dragon_Original_Data0186.jpg`, `Defect_Dragon_Original_Data0058.jpg`. Names use underscores and a zero-padded numeric identifier; `Original_Data` appears in the examples.
- **Mango:** class folders provide the labels. Filenames follow `image_0_<number>.jpg`, e.g. `image_0_1003.jpg`; no fruit/sample key, date, view, or acquisition condition is documented in the inspected files.
- **Avocado:** class folders use numbered prefixes: `1_unripe`, `2_semiripe`, `3_ripe`, `4_overripe`, `5_rotten`. Example: `T10_002__T10_d01_002_a_1.jpg`. The build script forms `fruit_id` as `<Storage Group>_<Sample padded to 3 digits>` and splits at fruit level. The filename also shows a storage token (`T10`, `T20`, or `Tam` in the summary), `d` plus a two-digit token, sample number, a letter (`a`/`b`), and numeric label; the inspected documentation does not define the meaning of `dNN` or `a`/`b`.
- **Papaya source export:** names look like `aug_0_a_frame_100_jpg.rf.<32-hex-id>.jpg` with a matching `.txt` label basename. The crop script parses `aug_<integer>_<video-token>_frame_<integer>`; `video-token` examples are `a`, `b`, `c`, `dd`. The `.rf.<hex>` suffix is present in the Roboflow export names. Crop names are `<video>_f<frame:03d>_aug<augmentation-index>_b<box-index>.jpg`, e.g. `a_f100_aug0_b0.jpg`; class folders carry the label.

The class folders and filenames encode some labels and apparent view/frame identifiers. Across the project, there is no documented capture calendar/date, angle protocol beyond banana filename view terms, or lighting-condition field. Do not infer those from folder names.

## CSV and annotation schemas

All CSV examples below are from existing headers and rows. CSV records are stored as text; types are described as apparent/derivable from values or the conversion script, not as enforced schema declarations. The manifests contain absolute source/destination paths from a different machine under `/Users/fakhridjmrs/...`; those path roots are not available as project contents here.

| File | Columns (in order) | Meaning / observable types |
|---|---|---|
| `avocado_hass_by_class/labels_matched.csv` | `src`, `dst`, `fruit_id`, `storage`, `label`, `class`, `split` | Paths and identifiers are strings; `fruit_id` example `T20_001`, `storage` string (`T10`/`T20`/`Tam`), `label` numeric class code serialized in CSV (1–5), class string, split `Training` or `Testing`. Each row links a source photo to its copied class-folder destination and fruit-level grouping. |
| `avocado_hass_by_class/unmatched.csv` | `src`, `reason` | Strings. Summary reports 12 unmatched entries, all `label_without_file`; rows are label/source stems without a matching image. |
| `papaya/papaya_by_class/crops.csv` | `src`, `dst`, `split`, `video`, `frame`, `aug`, `label`, `class`, `crop_w`, `crop_h` | Paths and category fields are strings; `frame`, `aug`, `crop_w`, `crop_h` are integer-like values (serialized as text); labels include `unripe`, `semiripe`, `ripe`, `rotten`; split is `Training`/`Testing`. One row maps one source detection box to a cropped image. |

Papaya annotation `.txt` files are YOLO-style rows: `class_id x_center y_center width height`, with one row per object and normalized decimal coordinates (example: `3 0.72734375 0.41015625 0.5453125 0.49375`). `data.yaml` says `nc: 4`, names in index order `['ripe', 'rotten', 'semiripe', 'unripe']`; the crop script uses that order and maps them to numbered folders (`1_unripe`, `2_semiripe`, `3_ripe`, `4_rotten`). Image and label files pair by basename, with the `.jpg` image and `.txt` label under parallel `images` and `labels` directories.

## Existing documentation and scripts

- `papaya/README.dataset.txt` identifies “Papaya Ripeness Detection,” links its Roboflow Universe dataset, and states CC BY 4.0.
- `papaya/README.roboflow.txt` says the export has 1,951 images, was exported March 27, 2026, uses a 640x640 stretch resize with EXIF orientation stripped, and reports no image augmentation applied by that export. It says “YOLOv8 are annotated in YOLO26 format” (preserved wording from file).
- `papaya/data.yaml` defines four classes and references `train/images`, `../valid/images`, and `test/images`; the current tree contains `train` and `test`, but no `valid` directory.
- `papaya/papaya_by_class/papaya_crops.py` describes converting annotated boxes to class-folder crops, with 10% padding around each box. Its summary records counts and source-video/class counts. It also records 191 `(video, frame)` pairs shared across train and test, states each class comes from a single video in both splits, and explicitly says Testing is **NOT** independent of Training. Video `b` is mostly semiripe with 4 frames labeled ripe; video `c` is mostly ripe with 3 frames labeled semiripe; those labels were retained as labeled.
- `avocado_hass_by_class/build_by_class.py` describes a Hass avocado ripening dataset to Create ML layout, with an 80:20 split by fruit, seed 42, stratified by storage group; it refers to a source workbook sheet `DATABASE` and source image directory. The script’s expected workbook and original image directory are not present here. `summary.txt` says 478 matched unique fruits and no `fruit_id` overlap between Training and Testing; it reports 12 unmatched labels without matching files and 14,710 copied images.
- No collection protocol or label definitions were found for the Banana, DragonFruit, or Mango collections beyond the folder/file labels themselves. No general project goal, planned roadmap, or future implementation plan is documented in the inspected files.

## Current state and limits of verification

The tree currently contains the class-organized JPEG collections, an avocado matched/unmatched manifest and build script, and a papaya detection export plus a class-crop derivative, manifest, summary and crop script. No model artifacts or training/evaluation code were found. The two scripts describe dataset-conversion workflows; their presence does not establish that they can be rerun without external source data/dependencies.

The papaya `data.yaml` references a validation directory that is absent from the current tree. CSV source/destination paths point to an external machine path. The root has no Git metadata, shared project README, or documented acquisition metadata. Per-image visual content, EXIF details, exact definitions of avocado `dNN` / `a` / `b`, and lighting setup were not inspected or verifiable from the available documents. Image contents were not individually reviewed.
