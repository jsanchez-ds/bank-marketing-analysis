# 📁 Data

## Download Instructions

The dataset is not included in this repository due to size. To download it:

1. Visit the [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
2. Download `bank-additional-full.csv`
3. Place it in this `data/` folder

Alternatively, download directly via Python:

```python
from ucimlrepo import fetch_ucirepo
bank_marketing = fetch_ucirepo(id=222)
df = bank_marketing.data.original
df.to_csv('data/bank-additional-full.csv', index=False)
```

## Dataset Description

- **Source**: Moro et al. (2014). UCI Machine Learning Repository.
- **Records**: 45,211
- **Features**: 16 input variables + 1 binary target (`y`)
- **Target**: Whether the client subscribed to a term deposit (yes/no)
