# Striatal feature validation

This report describes deterministic image-derived features. Acquisition metadata are not included in the primary feature model.

## Issues

- high_uptake_volume_mm3_rel_0_50: duplicate values with high_uptake_physical_volume_mm3
- number_of_connected_components_rel_0_50: duplicate values with number_of_connected_components
- roi_high_uptake_fraction: duplicate values with high_uptake_fraction_rel_0_50

## Feature statistics

| feature | min | max | mean | std | missing_count | unique_count | family |
| --- | --- | --- | --- | --- | --- | --- | --- |
| asymmetry_high_uptake_volume_mm3_abs | 0 | 145328.12 | 6784.5769 | 12036.903 | 0 | 680 | asymmetry |
| asymmetry_high_uptake_volume_mm3_normalized | 0 | 2 | 0.24582384 | 0.28371067 | 0 | 1361 | asymmetry |
| asymmetry_mean_abs | 1.4603138e-05 | 0.22997189 | 0.04361236 | 0.041677352 | 0 | 1361 | asymmetry |
| asymmetry_mean_normalized | 2.9973259e-05 | 0.83698323 | 0.12957651 | 0.13140793 | 0 | 1362 | asymmetry |
| asymmetry_p95_abs | 0 | 0.23385397 | 0.027001602 | 0.028938536 | 0 | 1314 | asymmetry |
| asymmetry_p95_normalized | 0 | 0.29655283 | 0.037225814 | 0.039713369 | 0 | 1314 | asymmetry |
| background_uptake | 0.021745611 | 0.50833333 | 0.22702552 | 0.097341153 | 0 | 1360 | background_ratio |
| bilateral_sbr_like | -0.034687764 | 8.4242214 | 0.89880208 | 0.99542272 | 0 | 1362 | background_ratio |
| bilateral_striatum_to_background | 0.96530983 | 9.4241754 | 1.8987964 | 0.99541897 | 0 | 1362 | background_ratio |
| component_volume_ratio | 1 | 14013.5 | 50.871708 | 513.2423 | 0 | 1320 | morphology |
| high_uptake_fraction_rel_0_40 | 0.0039141068 | 0.5917086 | 0.072337831 | 0.080266197 | 0 | 1362 | uptake |
| high_uptake_fraction_rel_0_50 | 0.0017656336 | 0.35351647 | 0.023879783 | 0.034882665 | 0 | 1358 | uptake |
| high_uptake_fraction_rel_0_60 | 0.00056568843 | 0.17388178 | 0.010294637 | 0.014203154 | 0 | 1354 | uptake |
| high_uptake_physical_volume_mm3 | 4218.75 | 695187.5 | 58849.28 | 64267.693 | 0 | 1204 | morphology |
| high_uptake_volume_mm3_rel_0_40 | 10703.125 | 1631562.5 | 190657.39 | 191747.27 | 0 | 1307 | morphology |
| high_uptake_volume_mm3_rel_0_50 | 4218.75 | 695187.5 | 58849.28 | 64267.693 | 0 | 1204 | morphology |
| high_uptake_volume_mm3_rel_0_60 | 1546.875 | 404265.62 | 25812.018 | 24327.019 | 0 | 1059 | morphology |
| high_uptake_voxel_count | 270 | 44492 | 3766.3539 | 4113.1323 | 0 | 1204 | morphology |
| largest_component_volume_mm3 | 2328.125 | 684265.62 | 39419.936 | 61284.927 | 0 | 1096 | morphology |
| left_anterior_uptake | 0.11965074 | 0.67408109 | 0.35871279 | 0.10353107 | 0 | 1362 | anterior_posterior |
| left_bounding_box_extent_x_mm | 0 | 80 | 34.498899 | 13.438396 | 0 | 32 | shape |
| left_bounding_box_extent_y_mm | 0 | 160 | 51.317915 | 26.882799 | 0 | 61 | shape |
| left_bounding_box_extent_z_mm | 0 | 120 | 37.714758 | 16.595703 | 0 | 46 | shape |
| left_compactness_like | 0 | 0.63650794 | 0.3351801 | 0.11484638 | 0 | 1340 | shape |
| left_elongation_ratio | 0 | 4.4158683 | 1.5143598 | 0.3138008 | 0 | 1362 | shape |
| left_high_uptake_volume_mm3 | 0 | 303453.12 | 27517.392 | 30448.701 | 0 | 1060 | morphology |
| left_lambda2_over_lambda1 | 0 | 0.97027697 | 0.4780913 | 0.15540921 | 0 | 1362 | shape |
| left_lambda3_over_lambda1 | 0 | 0.88056592 | 0.26493494 | 0.11151507 | 0 | 1362 | shape |
| left_max | 0.47353566 | 2 | 1.8585078 | 0.26866989 | 0 | 456 | left_right |
| left_mean | 0.11961973 | 0.59701061 | 0.34292822 | 0.088260466 | 0 | 1362 | left_right |
| left_p50 | 0.016665893 | 0.63044927 | 0.31844856 | 0.12662937 | 0 | 1358 | left_right |
| left_p75 | 0.1396588 | 0.78143907 | 0.51085798 | 0.11439841 | 0 | 1357 | left_right |
| left_p90 | 0.26952517 | 0.93114227 | 0.64153162 | 0.12735791 | 0 | 1359 | left_right |
| left_p95 | 0.30427524 | 1.03738 | 0.72053774 | 0.13773054 | 0 | 1359 | left_right |
| left_p99 | 0.3717301 | 1.8144022 | 1.0192796 | 0.20209717 | 0 | 1359 | left_right |
| left_p99_5 | 0.39137353 | 2 | 1.2750675 | 0.30555976 | 0 | 1334 | left_right |
| left_posterior_to_anterior | 0.22506988 | 2.251812 | 0.93701855 | 0.19884244 | 0 | 1362 | anterior_posterior |
| left_posterior_uptake | 0.060125895 | 0.65713388 | 0.32679993 | 0.087970777 | 0 | 1362 | anterior_posterior |
| left_principal_axis_length_1_mm | 0 | 181.85314 | 49.028458 | 24.860772 | 0 | 1362 | shape |
| left_principal_axis_length_2_mm | 0 | 117.63029 | 32.157306 | 13.108881 | 0 | 1362 | shape |
| left_principal_axis_length_3_mm | 0 | 70.064197 | 23.22864 | 8.7061018 | 0 | 1362 | shape |
| left_sbr_like | -0.063296794 | 11.567242 | 0.77665779 | 0.87979826 | 0 | 1362 | background_ratio |
| left_std | 0.089696849 | 0.3593908 | 0.24479863 | 0.051596682 | 0 | 1362 | left_right |
| left_striatum_to_background | 0.9367008 | 12.567196 | 1.7766521 | 0.87979469 | 0 | 1362 | background_ratio |
| maximum_side_uptake | 0.13996948 | 0.62620914 | 0.38112773 | 0.079721886 | 0 | 1362 | left_right |
| mean_bilateral_uptake | 0.13704927 | 0.58073571 | 0.35932155 | 0.078980038 | 0 | 1362 | left_right |
| mean_posterior_to_anterior | 0.22620146 | 1.9027608 | 0.92975761 | 0.1853555 | 0 | 1362 | anterior_posterior |
| minimum_posterior_to_anterior | 0.22506988 | 1.7229851 | 0.90602877 | 0.178099 | 0 | 1362 | anterior_posterior |
| minimum_side_to_background | 0.89533453 | 7.7291335 | 1.746064 | 0.82302685 | 0 | 1362 | background_ratio |
| minimum_side_uptake | 0.11961973 | 0.57962775 | 0.33751537 | 0.083597934 | 0 | 1362 | left_right |
| number_of_connected_components | 1 | 142 | 19.606461 | 20.826482 | 0 | 97 | morphology |
| number_of_connected_components_rel_0_40 | 1 | 250 | 29.5 | 26.619863 | 0 | 118 | morphology |
| number_of_connected_components_rel_0_50 | 1 | 142 | 19.606461 | 20.826482 | 0 | 97 | morphology |
| number_of_connected_components_rel_0_60 | 1 | 111 | 7.6585903 | 12.320733 | 0 | 67 | morphology |
| right_anterior_uptake | 0.1084149 | 0.68368667 | 0.39478137 | 0.092878517 | 0 | 1362 | anterior_posterior |
| right_bounding_box_extent_x_mm | 10 | 80 | 38.797724 | 13.737563 | 0 | 29 | shape |
| right_bounding_box_extent_y_mm | 12.5 | 160 | 52.668869 | 26.909269 | 0 | 58 | shape |
| right_bounding_box_extent_z_mm | 5 | 120 | 38.353524 | 16.137465 | 0 | 44 | shape |
| right_compactness_like | 0.055769231 | 0.59902597 | 0.311533 | 0.10246853 | 0 | 1334 | shape |
| right_elongation_ratio | 1.0125433 | 3.2210935 | 1.5011291 | 0.28070857 | 0 | 1362 | shape |
| right_high_uptake_volume_mm3 | 578.125 | 391734.38 | 31331.888 | 35017.543 | 0 | 1095 | morphology |
| right_lambda2_over_lambda1 | 0.096381425 | 0.97537766 | 0.48346436 | 0.15244535 | 0 | 1362 | shape |
| right_lambda3_over_lambda1 | 0.011788294 | 0.77467632 | 0.28098851 | 0.10894001 | 0 | 1362 | shape |
| right_max | 0.43648151 | 2 | 1.8674814 | 0.26134236 | 0 | 443 | left_right |
| right_mean | 0.13390464 | 0.62620917 | 0.37571489 | 0.077251665 | 0 | 1362 | left_right |
| right_p50 | 0.017587472 | 0.64569959 | 0.37155838 | 0.10056431 | 0 | 1358 | left_right |
| right_p75 | 0.1464185 | 0.78788659 | 0.53663163 | 0.10833709 | 0 | 1359 | left_right |
| right_p90 | 0.27208907 | 0.93349175 | 0.65964547 | 0.12634538 | 0 | 1359 | left_right |
| right_p95 | 0.30548631 | 1.0413339 | 0.73865463 | 0.13738339 | 0 | 1359 | left_right |
| right_p99 | 0.36407487 | 1.9484438 | 1.0563699 | 0.21013764 | 0 | 1358 | left_right |
| right_p99_5 | 0.38117711 | 2 | 1.3189168 | 0.31235592 | 0 | 1336 | left_right |
| right_posterior_to_anterior | 0.22733305 | 1.7229851 | 0.92249666 | 0.17790547 | 0 | 1362 | anterior_posterior |
| right_posterior_uptake | 0.084777333 | 0.70793372 | 0.35637498 | 0.080490149 | 0 | 1362 | anterior_posterior |
| right_principal_axis_length_1_mm | 16.850913 | 189.39164 | 50.429256 | 24.768066 | 0 | 1362 | shape |
| right_principal_axis_length_2_mm | 10.349817 | 111.95507 | 33.458569 | 13.622457 | 0 | 1362 | shape |
| right_principal_axis_length_3_mm | 3.9763131 | 75.512697 | 24.726944 | 8.9377235 | 0 | 1362 | shape |
| right_sbr_like | -0.10466209 | 11.200101 | 1.0140692 | 1.1540177 | 0 | 1362 | background_ratio |
| right_std | 0.088659222 | 0.37296544 | 0.24337914 | 0.052791218 | 0 | 1362 | left_right |
| right_striatum_to_background | 0.89533453 | 12.200063 | 2.0140635 | 1.154014 | 0 | 1362 | background_ratio |
| roi_high_uptake_fraction | 0.0017656336 | 0.35351647 | 0.023879783 | 0.034882665 | 0 | 1358 | uptake |
| roi_max | 0.47353566 | 2 | 1.8910144 | 0.24128436 | 0 | 372 | uptake |
| roi_mean | 0.13704927 | 0.58073571 | 0.35969672 | 0.078695267 | 0 | 1362 | uptake |
| roi_p50 | 0.047040567 | 0.63034603 | 0.35046498 | 0.10603583 | 0 | 1358 | uptake |
| roi_p75 | 0.14282995 | 0.77787764 | 0.52647629 | 0.10854278 | 0 | 1357 | uptake |
| roi_p90 | 0.27078153 | 0.9222021 | 0.65206639 | 0.12575243 | 0 | 1359 | uptake |
| roi_p95 | 0.30444139 | 1.0165286 | 0.73082058 | 0.13645557 | 0 | 1359 | uptake |
| roi_p99 | 0.37071904 | 1.8409831 | 1.0389841 | 0.19839608 | 0 | 1359 | uptake |
| roi_p99_5 | 0.38627323 | 2 | 1.3010741 | 0.30083241 | 0 | 1339 | uptake |
| roi_positive_fraction | 0.045562744 | 1 | 0.91627691 | 0.14566622 | 0 | 927 | uptake |
| roi_std | 0.089198292 | 0.36631743 | 0.24591647 | 0.052392983 | 0 | 1362 | uptake |
| second_largest_component_volume_mm3 | 0 | 73140.625 | 11381.138 | 9121.6484 | 0 | 913 | morphology |
