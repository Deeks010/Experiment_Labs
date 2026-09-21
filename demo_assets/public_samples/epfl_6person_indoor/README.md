# EPFL six-person indoor sample

This is a research sample downloaded from the official EPFL CVLab page on 2026-09-20.

It contains four synchronized indoor camera videos, each 360x288 at 25 FPS and about
118 seconds long. Up to six people walk through the same room. The scene has fixed
chairs, tables, a carpet, walls and equipment that can be mapped as static anchors.

Files:

- `6p-c0.avi` through `6p-c3.avi`: synchronized camera views.
- `calibration-6p.txt`: camera homographies for projecting image points to the floor.
- `gt_lab_6p.txt`: labelled ground-plane positions for six people at regular frames.

Source: https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/

## What this can test

1. Whether the detector finds people in each camera.
2. Whether one camera keeps a local ID during movement and occlusion.
3. Whether the same person can be matched across camera views.
4. Whether image footpoints can be projected onto one shared floor map.
5. Whether mapped static anchors and floor areas give useful movement summaries.

## What it cannot test

It is not a factory. It has no production task, pallet, machine workflow or real
operational loss. It can validate the measurement layer, not the business report.

Use the accompanying ground truth as the accuracy check. Do not treat our tracker
output as correct merely because the visual preview looks smooth.
