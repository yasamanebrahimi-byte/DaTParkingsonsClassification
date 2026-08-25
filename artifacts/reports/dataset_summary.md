# Dataset summary

- Scans: 1362
- Normal: 615
- Pathologic: 747
- Pathologic fraction: 0.5485

## Shapes

          count        mean        std    min    25%    50%    75%    max
shape_x  1362.0  171.944200  60.195883  118.0  128.0  128.0  256.0  256.0
shape_y  1362.0  171.944200  60.195883  118.0  128.0  128.0  256.0  256.0
shape_z  1362.0  142.712922  86.338477   33.0   55.0  128.0  256.0  256.0

## Spacing (mm)

            count      mean       std      min     25%   50%      75%     max
spacing_x  1362.0  2.758013  0.715570  1.37222  2.3976  2.46  3.59095  4.4196
spacing_y  1362.0  2.758013  0.715570  1.37222  2.3976  2.46  3.59095  4.4196
spacing_z  1362.0  2.824011  0.683351  1.50000  2.3976  2.46  3.89537  4.4196

## Orientations

RAS    1362

## Stored dtypes

float32    1362

## Intensity and foreground

                   count         mean           std        min        25%        50%          75%           max
min_intensity     1362.0    -0.085169      1.058925 -27.000000   0.000000    0.00000     0.000000      0.000000
max_intensity     1362.0  6713.250367  12746.171442   3.000000  69.000000  255.00000  9422.750000  65466.000000
median_nonzero    1362.0   552.195668   1804.801163   1.000000   3.000000    8.00000   106.000000  14146.000000
p99_5_nonzero     1362.0  2785.217173   6065.293107   2.000000  24.250000   80.00000  3739.000000  38451.200000
nonzero_fraction  1362.0     0.268516      0.244011   0.012789   0.037239    0.17625     0.436596      0.962326

## Notes

The acquisition geometry is heterogeneous: spacing, matrix shape, physical extent, and intensity scale vary across scans. The model therefore uses affine-aware resampling and per-scan normalization. Metadata-only leakage results are reported separately.
