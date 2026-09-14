# Push/fold policy comparison

All four seats use each named policy. Net-point differences describe paired table outcomes, not focal-player EV. Payments include dealer-aware ron and tsumo, riichi deposits, multiple-ron settlement, and exhaustive-draw transfers; ippatsu and honba are excluded.

## Policy summaries

| policy | tables | ron endings | multiple ron | tsumo | exhaustive draw | riichi/table | mean ron | mean tsumo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all_push | 20000 | 0.6452 | 0.0120 | 0.2450 | 0.1098 | 1.5695 | 5925.0 | 7490.9 |
| full_fold | 20000 | 0.0974 | 0.0001 | 0.3568 | 0.5458 | 0.9637 | 6049.5 | 7605.9 |
| mixed_mean | 20000 | 0.2464 | 0.0006 | 0.3685 | 0.3851 | 1.3202 | 6080.4 | 7649.8 |
| mixed_minimax | 20000 | 0.2480 | 0.0008 | 0.3673 | 0.3847 | 1.3201 | 6079.0 | 7642.7 |

## All-tile danger by declared opponents

### all_push

| opponents | candidate rate (SE) | chosen rate (SE) |
| ---: | ---: | ---: |
| 1 | 0.0626 (0.0003) | 0.0457 (0.0005) |
| 2 | 0.1197 (0.0007) | 0.0872 (0.0013) |
| 3 | 0.1757 (0.0023) | 0.1309 (0.0049) |

### full_fold

| opponents | candidate rate (SE) | chosen rate (SE) |
| ---: | ---: | ---: |
| 1 | 0.0734 (0.0003) | 0.0044 (0.0001) |
| 2 | — | — |
| 3 | — | — |

### mixed_mean

| opponents | candidate rate (SE) | chosen rate (SE) |
| ---: | ---: | ---: |
| 1 | 0.0713 (0.0003) | 0.0129 (0.0002) |
| 2 | 0.1320 (0.0010) | 0.0143 (0.0006) |
| 3 | 0.1867 (0.0070) | 0.0121 (0.0043) |

### mixed_minimax

| opponents | candidate rate (SE) | chosen rate (SE) |
| ---: | ---: | ---: |
| 1 | 0.0714 (0.0003) | 0.0129 (0.0002) |
| 2 | 0.1319 (0.0010) | 0.0151 (0.0006) |
| 3 | 0.1864 (0.0071) | 0.0153 (0.0048) |

## Paired differences

Each value is right minus left, with a paired-table standard error.

### full_fold__minus__all_push

Paired tables: 20000

| measure | mean difference | SE |
| --- | ---: | ---: |
| seat 0 net points | -387.1550 | 35.8990 |
| seat 1 net points | 102.0550 | 30.3658 |
| seat 2 net points | -42.5200 | 29.8716 |
| seat 3 net points | -40.0800 | 29.5341 |
| ron indicator | -0.5479 | 0.0037 |
| tsumo indicator | 0.1118 | 0.0032 |
| exhaustion indicator | 0.4361 | 0.0036 |
| riichi declarations | -0.6058 | 0.0051 |

### mixed_mean__minus__all_push

Paired tables: 20000

| measure | mean difference | SE |
| --- | ---: | ---: |
| seat 0 net points | -355.2450 | 33.6009 |
| seat 1 net points | 60.2300 | 29.4007 |
| seat 2 net points | -26.7200 | 28.5763 |
| seat 3 net points | 20.0350 | 28.4317 |
| ron indicator | -0.3989 | 0.0037 |
| tsumo indicator | 0.1235 | 0.0030 |
| exhaustion indicator | 0.2753 | 0.0033 |
| riichi declarations | -0.2493 | 0.0045 |

### mixed_minimax__minus__all_push

Paired tables: 20000

| measure | mean difference | SE |
| --- | ---: | ---: |
| seat 0 net points | -352.3400 | 33.7185 |
| seat 1 net points | 57.8550 | 29.4986 |
| seat 2 net points | -18.6450 | 28.5823 |
| seat 3 net points | 12.2800 | 28.3722 |
| ron indicator | -0.3972 | 0.0037 |
| tsumo indicator | 0.1222 | 0.0030 |
| exhaustion indicator | 0.2750 | 0.0033 |
| riichi declarations | -0.2494 | 0.0045 |

### mixed_mean__minus__full_fold

Paired tables: 20000

| measure | mean difference | SE |
| --- | ---: | ---: |
| seat 0 net points | 31.9100 | 25.1038 |
| seat 1 net points | -41.8250 | 20.6735 |
| seat 2 net points | 15.8000 | 19.9411 |
| seat 3 net points | 60.1150 | 19.6419 |
| ron indicator | 0.1490 | 0.0027 |
| tsumo indicator | 0.0118 | 0.0020 |
| exhaustion indicator | -0.1608 | 0.0027 |
| riichi declarations | 0.3564 | 0.0036 |

### mixed_minimax__minus__full_fold

Paired tables: 20000

| measure | mean difference | SE |
| --- | ---: | ---: |
| seat 0 net points | 34.8150 | 25.2174 |
| seat 1 net points | -44.2000 | 20.7266 |
| seat 2 net points | 23.8750 | 19.9335 |
| seat 3 net points | 52.3600 | 19.6585 |
| ron indicator | 0.1507 | 0.0027 |
| tsumo indicator | 0.0104 | 0.0020 |
| exhaustion indicator | -0.1611 | 0.0027 |
| riichi declarations | 0.3564 | 0.0036 |

### mixed_minimax__minus__mixed_mean

Paired tables: 20000

| measure | mean difference | SE |
| --- | ---: | ---: |
| seat 0 net points | 2.9050 | 5.8256 |
| seat 1 net points | -2.3750 | 4.8044 |
| seat 2 net points | 8.0750 | 4.4246 |
| seat 3 net points | -7.7550 | 4.5800 |
| ron indicator | 0.0017 | 0.0006 |
| tsumo indicator | -0.0013 | 0.0004 |
| exhaustion indicator | -0.0004 | 0.0005 |
| riichi declarations | -0.0001 | 0.0001 |
