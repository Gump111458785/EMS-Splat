# Occlusion protocols

The built-in benchmark degrades precomputed 2D observation heatmaps and their
validity. It does not rerun a detector on masked RGB unless an external native
RGB detector pipeline is explicitly used.

## Rectangle

One contiguous rectangle covers a configured fraction of the observation
plane. Ratios used in the paper are 0.2, 0.4, and 0.6.

## Random block

The same total area is divided across four independently placed blocks.
Ratios used in the paper are 0.2, 0.4, and 0.6.

## Body part

Semantic masks remove both arms, both legs, or the torso. A body-part setting
is not an area-ratio experiment and must not be labelled as ratio 0.

## Seeds

`test.occlusion.seed` controls mask sampling. `training.seed` controls
per-scene optimization randomness. Neither is a network-training seed.

## Public/native protocols

Manifest-backed public VOC masks and native RGB detector experiments are
separate protocols. Do not merge their values with observation-space results
without an interface label.

## Public release

The canonical controlled conditions and mask-seed policy are distributed in
`data/protocols/ems_occlusion_protocol_v1.csv`. This metadata is
redistributable because it contains no base-dataset images, annotations,
camera parameters or detector output. Licensing and generated-data boundaries
are documented in `docs/DATASET_CARD.md`.
