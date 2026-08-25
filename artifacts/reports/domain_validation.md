# Domain-aware validation

Domain-aware CV holds acquisition families out of each validation fold. It is a robustness diagnostic and does not replace canonical IID StratifiedKFold.

## Standard vs domain-aware OOF

| Validation | Log Loss | AUROC | Brier |
| --- | ---: | ---: | ---: |
| Standard Stratified CV | pending OOF predictions | pending | pending |
| Domain-Aware CV | pending OOF predictions | pending | pending |

Neural-model OOF predictions were not supplied, so this implementation report does not fabricate performance values.

## Domain-fold distribution

| fold | sample_count | normal_count | pathologic_count | pathologic_fraction | number_of_domains | domain_names_counts |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 243 | 115 | 128 | 0.526749 | 1 | domain_05:243 |
| 1 | 65 | 37 | 28 | 0.430769 | 1 | domain_01:65 |
| 2 | 219 | 111 | 108 | 0.493151 | 2 | domain_04:39, domain_06:180 |
| 3 | 255 | 107 | 148 | 0.580392 | 1 | domain_00:255 |
| 4 | 580 | 245 | 335 | 0.577586 | 3 | domain_02:88, domain_03:32, domain_07:460 |

## Domain-level results

No domain results.

