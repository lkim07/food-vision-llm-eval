Paired bootstrap over images, 2000 resamples, 95% CI. A difference is significant if its CI excludes 0.
macro_within_20: higher is better. macro_PMAE: lower is better.

## MM-Food-100K (GPT-4o labels)

| model                 | macro_within_20_CI95   | macro_PMAE_CI95   |
|:----------------------|:-----------------------|:------------------|
| gpt-4o-mini           | [33.7, 47.5]           | [27.6, 42.3]      |
| gemini-2.5-flash-lite | [21.0, 32.1]           | [66.2, 101.2]     |
| claude-haiku-4.5      | [27.7, 40.3]           | [54.0, 86.6]      |

| metric          | comparison                               |   mean_diff |   ci95_low |   ci95_high | significant   |
|:----------------|:-----------------------------------------|------------:|-----------:|------------:|:--------------|
| macro_within_20 | gpt-4o-mini - gemini-2.5-flash-lite      |        14.2 |        6.7 |        21.2 | True          |
| macro_within_20 | gpt-4o-mini - claude-haiku-4.5           |         6.5 |       -2.2 |        14.9 | False         |
| macro_within_20 | gemini-2.5-flash-lite - claude-haiku-4.5 |        -7.6 |      -15.8 |         0.5 | False         |
| macro_PMAE      | gpt-4o-mini - gemini-2.5-flash-lite      |       -48.4 |      -65.6 |       -33.7 | True          |
| macro_PMAE      | gpt-4o-mini - claude-haiku-4.5           |       -35.8 |      -52.9 |       -19.8 | True          |
| macro_PMAE      | gemini-2.5-flash-lite - claude-haiku-4.5 |        12.6 |       -6   |        29.6 | False         |

## Nutrition5k (measured labels)

| model                 | macro_within_20_CI95   | macro_PMAE_CI95   |
|:----------------------|:-----------------------|:------------------|
| gpt-4o-mini           | [16.8, 26.5]           | [51.7, 71.5]      |
| gemini-2.5-flash-lite | [18.5, 30.5]           | [45.2, 61.6]      |
| claude-haiku-4.5      | [13.5, 22.3]           | [49.7, 67.4]      |

| metric          | comparison                               |   mean_diff |   ci95_low |   ci95_high | significant   |
|:----------------|:-----------------------------------------|------------:|-----------:|------------:|:--------------|
| macro_within_20 | gpt-4o-mini - gemini-2.5-flash-lite      |        -2.9 |       -8.8 |         3.5 | False         |
| macro_within_20 | gpt-4o-mini - claude-haiku-4.5           |         3.7 |       -2   |         9.7 | False         |
| macro_within_20 | gemini-2.5-flash-lite - claude-haiku-4.5 |         6.6 |        0.3 |        13.2 | True          |
| macro_PMAE      | gpt-4o-mini - gemini-2.5-flash-lite      |         8.1 |        1.6 |        14.6 | True          |
| macro_PMAE      | gpt-4o-mini - claude-haiku-4.5           |         2.9 |       -5.6 |        11.2 | False         |
| macro_PMAE      | gemini-2.5-flash-lite - claude-haiku-4.5 |        -5.2 |      -12.8 |         2.1 | False         |

