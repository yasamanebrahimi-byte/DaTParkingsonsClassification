# Leave-one-domain-out split manifest

- Minimum validation-domain size: 30
- Eligible validation domains: 8
- Each row belongs to one diagnostic run identified by `validation_domain`; this is not a standard single-fold assignment file.

| validation_domain | sample_count | normal_count | pathologic_count | pathologic_fraction |
| --- | ---: | ---: | ---: | ---: |
| domain_00 | 255 | 107 | 148 | 0.580392 |
| domain_01 | 65 | 37 | 28 | 0.430769 |
| domain_02 | 88 | 37 | 51 | 0.579545 |
| domain_03 | 32 | 21 | 11 | 0.343750 |
| domain_04 | 39 | 18 | 21 | 0.538462 |
| domain_05 | 243 | 115 | 128 | 0.526749 |
| domain_06 | 180 | 93 | 87 | 0.483333 |
| domain_07 | 460 | 187 | 273 | 0.593478 |

For each run, train on rows with `split=train` and evaluate on rows with `split=validation`.
