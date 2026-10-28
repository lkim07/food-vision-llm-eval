## Processed foods (MM-Food-100K)

| model                 |   n |   dish_name_% |   cooking_method_% |   portion_within_20_% |   calories_within_20_% |   protein_within_20_% |   fat_within_20_% |   carbs_within_20_% |   calories_median_bias_% |   protein_median_bias_% |   fat_median_bias_% |   carbs_median_bias_% |   few_shot_copies |   calories_exact_label_match_% |
|:----------------------|----:|--------------:|-------------------:|----------------------:|-----------------------:|----------------------:|------------------:|--------------------:|-------------------------:|------------------------:|--------------------:|----------------------:|------------------:|-------------------------------:|
| GPT-4o mini           | 400 |          54.5 |               88.2 |                  23.5 |                   51.7 |                  54.8 |              41   |                44.3 |                    -16.7 |                   -12.5 |               -20   |                 -16.7 |                 2 |                           27.3 |
| Gemini 2.0 Flash-Lite | 400 |          42   |               86.5 |                  27.8 |                   52.5 |                  58   |              37.2 |                49   |                      0   |                     0   |                20   |                   0   |                 1 |                           18   |
| Claude 3 Haiku        | 399 |          31.3 |               73.4 |                  25.4 |                   32.8 |                  45.1 |              30.6 |                27.9 |                    -25   |                   -14.3 |               -26.7 |                 -26.7 |                20 |                            8.8 |

## Simple foods (fruit & vegetable dataset)

| model                 |   n |   dish_name_% |
|:----------------------|----:|--------------:|
| GPT-4o mini           | 100 |            92 |
| Gemini 2.0 Flash-Lite | 100 |            94 |
| Claude 3 Haiku        | 100 |            88 |
