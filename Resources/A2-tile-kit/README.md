# How the Assignment 2 dataset was made — and how to make your own

Assignment 2 gives you 64 × 64 tiles of the 1958 air photos, each labelled woodland, field,
edge or built. Nobody handed us those tiles: they were cut, chosen and labelled by hand, the
same way your team will need to make training data for a project. This folder explains how the
A2 set was built, what went wrong along the way, and gives you the same tools for **your own
image sets**.

| File | What it is |
|---|---|
| `README.md` | This guide. |
| `tilekit.py` | Cut tiles from your image set, make a labelling web page, turn the labels into a train/val/test file, and map a whole site with a trained model. Needs only `numpy` and `Pillow` (plus `torch` for mapping). |
| `tile_kit_demo.ipynb` | The whole workflow in Colab, start to finish, on one of your sites. |

`tilekit.py` contains **no network or training code**: plug in the network you build in
Assignment 2, or a pretrained one (L14, AL8).

**Most useful for:** Option 4 (Charlottetown urban growth), Option 8 (building identification)
and Option 1 (farm consolidation). See *Recipes* below. (Option 7, forest composition, would
fit too, but no team took it this term.)

---

## 1. How the A2 dataset was built

**Source.** The 1958 provincial orthomosaics: one huge image per county at 1.16 m per pixel.

**Tile size.** Each A2 tile is 128 × 128 source pixels averaged down 2 × 2 to 64 × 64, so one
tile covers about **148 m × 148 m** at 2.3 m per pixel. That is big enough to hold a farmyard or
a field edge, and small enough that most tiles are mostly one thing.

**Choosing tiles.** A purely random sample would have been almost all field and woodland:
buildings cover a tiny fraction of 1958 PEI. So the 1,439 candidate tiles came from three pools:

| Pool | Tiles | Why |
|---|---|---|
| Random tiles from the three counties | 420 | an honest picture of the landscape |
| Tiles crossed by streets and lanes (from a modern road map) | 300 | more farmyards and hamlets |
| Tiles within about 1 km of 40 towns and villages | 719 | enough *built* examples to learn from |

Rules applied while sampling:
- No two tiles came from the same 300 m × 300 m block, and tiles near towns were spaced about
  300 m apart, so near-duplicate neighbours rarely land on both sides of the train/test split.
- Tiles touching a gap in the mosaic were thrown out.

**A class we dropped.** The plan had a *water* class. But the historical mosaics were clipped
to the land, so in 1935–1968 every pond, river and bay is **pure black**. A network would learn
"black = water" and nothing else. Water was dropped, and *edge* (a mix of field and woodland)
took its place.

**Labelling.** The instructor labelled every tile in a web page. That page is the model for the
one `tilekit.py` makes. Two things made it workable:

- **Context.** On its own, a 148 m grey square is often unreadable. Next to each tile, the page
  showed the surrounding area at three times the width, at full resolution, plus the 2020 colour
  photo of the same spot. The 2020 photo shows where things are, not what they were in 1958, so
  labels always follow the 1958 photo.
- **Written rules, and a skip key.** For example: woodland means trees cover roughly ¾ of the
  tile; any building makes a tile *built*. Tiles that fit no rule were skipped:
  - forestry ink marks;
  - abandoned fields growing back to scrub;
  - marsh;
  - gaps.

  About 1 in 5 tiles was skipped. That is normal, and you should report it.

Labelling went at about 500 tiles per half hour once the rules were settled.

**Splitting.** The **test set is every tile from Kings County**: a place the network never saw
during training. Training and validation tiles come from Prince and Queens. A random split would
put tiles from the same village in both train and test, and the score would look better than it
deserves. The normalisation mean and standard deviation come from the **training tiles only**.

## 2. What we would tell a team starting now

1. **Write your class rules before you label**, and keep them in your repository. Your poster
   evaluators will ask what "urban" or "building" meant.
2. **Rare classes need targeted sampling.** Random sampling buries them. `tilekit` has a
   `texture` strategy that spreads tiles across smooth and busy areas. Busy tiles hold most of
   the buildings and hedgerows.
3. **Black is not always "no data".** In 1935, 1958 and 1968, black is also water. Deep shadow
   can be black in any year.
4. **Label about 50 tiles twice**, by two teammates or by yourself a week apart, and report the
   agreement. If you agree with yourself 85% of the time, a model at 85% is as good as your
   labels allow.
5. **Split by place**, or by year, depending on your question. Never split at random.
6. **Commit your labels files.** Your repository history is part of your grade, and labels are
   real work.
7. **Look before you trust.** Run an occlusion map (A2, Part 6) on a few tiles. Make sure the
   network isn't keying on ink marks, mosaic seams or the black water mask.

## 3. Using the kit on your image set

Your project image sets are folders of PNGs, one per year, all on the same pixel grid (see the
`info.txt` in each site folder). In Colab, open `tile_kit_demo.ipynb`. On your own computer:

```bash
# 1. cut 400 tiles from one year of your site, with context from every year
python tilekit.py cut --sites data/urban_growth/shared_both_groups/charlottetown \
                      --year 1958 --size 64 --n 400 --strategy texture --out tiles_1958

# 2. make the labelling page, open tiles_1958/labeller.html in a browser, label, Export labels
python tilekit.py labeller --tiles tiles_1958 --classes urban,agriculture,vegetated --title "1958 Charlottetown"

# 3. after labelling (one labels_<name>.json per teammate)
python tilekit.py agreement --labels tiles_1958/labels_*.json
python tilekit.py dataset --tiles tiles_1958 --labels tiles_1958/labels_*.json \
                          --classes urban,agriculture,vegetated --test-sites stratford --out urban_1958.npz
```

The `.npz` file has the same layout as `A2_tiles_1958.npz`: `X_train`, `y_train`, `X_val`, …,
`classes`, `mean`, `std`. Your A2 notebook code works on it almost unchanged; change the number
of classes in the last layer.

The labelling page holds every image inside one HTML file: about 50 KB per tile with four context
years. Keep it under a few hundred tiles per page, or pass `--context 1958,2020` to show fewer years.

To map a whole site with a trained model, use `tilekit.sliding_map(predict, image, size, stride)`.
It returns a grid of predicted classes, one cell per `stride` pixels. Do this for every year and
you have a change map.

## 4. Recipes

### Option 4 — Charlottetown urban growth
- **Classes:** `urban, agriculture, vegetated`, plus `water` only where you can see it (below).
- **Water:** in 1935–1968 water is black. Because the years are pixel-aligned, you can take
  water from **2000** (where it is visible) and apply that mask to the historical years. Label
  three land classes in the old photos.
- **Tiles:** `--size 64` at 1 m (64 m squares) keeps a city block and a field apart.
  `--factor 2` gives 128 m squares, which are faster but coarser.
- **Label each year separately.** A tile that is a field in 1958 may be a subdivision in 2000.
  Expect lower accuracy on the grey years than on 2020: the project asks you to measure that gap.
- **Split:** train on Charlottetown; test on your own group's site (Stratford or Cornwall).
- **Output:** run `sliding_map` on each year, and measure the urban area per year from the map.
  That's your growth curve. The classical method you compare against works on the same grid.

### Option 8 — Building identification (0.5 m, 1935 and 1968)
- **What tiles can and can't do:** a tile classifier answers *is there a building in this
  square?*, not *where exactly is its outline?* Use it for three things:
  - finding candidates quickly;
  - a deep-learning baseline you can score per tile against your classical detector and SAM;
  - checking where each method fails.
- **Classes:** `building, no_building`. Or `house, barn_or_large, no_building` if you want to
  study barns disappearing.
- **Tiles:** at 0.5 m a farmhouse is about 25 × 35 px, so `--size 48` (24 m) fits one
  building with some yard.
- **Sampling:** use `--strategy texture`. Most random tiles are empty fields.
- **Context:** cut context for both 1935 and 1968 (`--context 1935,1968`). Seeing the same spot
  in both years makes labelling and change-spotting much easier.
- **Mask R-CNN needs outlines, not tile labels.** Use the tiles to choose where to draw: outline
  buildings in your positive tiles with a polygon tool. The VGG Image Annotator (VIA) runs
  offline in a browser and exports JSON.

### Option 1 — Farm consolidation
- **Classes:** `field_interior, boundary` (hedgerow, fence line, lane), `woodland, built`.
- **Tiles:** `--size 32` to `48` at 1 m, so a boundary tile is mostly boundary.
- **Measure:** run `sliding_map` on each year and use **boundary density** (the share of
  boundary cells) as a consolidation measure: fewer, larger fields mean fewer boundaries.
  It's a deep-learning method you can set beside your classical edge-based segmentation, as
  the option requires.
- **Split by year:** train on two years, test on a third. Brightness and film change between
  years, and that is the generalisation your report needs to discuss.

## 5. Honest evaluation checklist

Put these in your report (the evaluation-honesty criterion is a quarter of the report grade):

- [ ] class definitions, and how many tiles of each you labelled
- [ ] how many you skipped, and why
- [ ] agreement on double-labelled tiles
- [ ] how you split (by site or year), and why
- [ ] accuracy per class, not only overall, plus a confusion matrix
- [ ] the accuracy on the held-out site or year next to validation accuracy, and an explanation of any gap
- [ ] one occlusion or failure example that shows what the model relies on

---

<sub>Built for MCS 3950, Fall 2026. The A2 tiles come from the 1958 provincial orthomosaics.
Building the A2 set needed georeferenced source mosaics that are not part of your image sets,
so those exact scripts aren't included; `tilekit.py` does the same steps on the image sets you
have.</sub>
