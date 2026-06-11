# Solar Power Generation Forecasting: UNISOLAR Dataset

Hourly solar generation forecasting across 42 PV sites using XGBoost, LightGBM, and Random Forest. The core question this notebook tries to answer is how much a physics-informed feature set (pvlib clearsky irradiance, solar position, thermal interactions) actually improves over a raw weather baseline, and whether those gains hold up when you run the same approach across every site in the dataset, not just the one you tuned on.

---

## Dataset

Three CSV files, all expected under `/content/drive/MyDrive/UNISOLAR_Data/`:

| File | Contents |
|---|---|
| `Solar_Energy_Generation.csv` | Per-site generation readings with `CampusKey`, `SiteKey`, and `Timestamp` |
| `Weather_Data_reordered_all.csv` | Matched weather observations (temperature, humidity, wind, dew point) |
| `Solar_Site_Details.csv` | Site metadata including `lat` and `Lon` per `SiteKey` |

The raw data is at 15-minute resolution. The notebook resamples it to hourly using `.resample('1h').mean()` and then multiplies `SolarGeneration` by 4 to convert those averaged 15-minute readings into kWh-equivalent hourly values. If your source data is already hourly, drop that multiplier.

Initial model development uses a single site (CampusKey=2, SiteKey=1). The multisite validation loop at the end iterates over all 42 unique `SiteKey` values in the full merged dataframe.

---

## Setup

The notebook runs on Google Colab. A T4 GPU is recommended, because the multisite loop uses `device='cuda'` in the XGBoost Optuna calls, and running 42 sites × 20 trials each on CPU will take considerably longer.

Two libraries aren't in the default Colab environment:

```bash
pip install pvlib
pip install optuna
```

Everything else (`xgboost`, `lightgbm`, `scikit-learn`, `pandas`, `numpy`, `shap`, `matplotlib`, `seaborn`) is already available.

Mount your Drive and confirm the three CSVs are at the expected path before running any data cells.

---

## Pipeline Overview

**1. Load and merge**

The three CSVs are joined on `CampusKey` + `Timestamp` (generation ↔ weather) and then `CampusKey` + `SiteKey` (merged ↔ site details) to attach lat/lon. Weather columns with gaps are filled with linear interpolation, then forward/backward filled for any remaining edge nulls.

**2. Two parallel experiments**

All models are trained and evaluated on two distinct feature sets:

- **Baseline (7 features):** The raw weather columns (`ApparentTemperature`, `AirTemperature`, `DewPointTemperature`, `RelativeHumidity`, `WindSpeed`, `WindDirection`) plus `hour` as a plain integer.
- **Engineered (24 features):** Everything above, extended with pvlib-derived solar physics, lag features, rolling statistics, thermal interaction terms, and cyclical time encodings (described below).

**3. Train/test split**

An 80/20 chronological split, earlier data trains, later data tests. No shuffling. Cross-validation inside Optuna also uses `TimeSeriesSplit(n_splits=3)` for the same reason: randomly splitting time series causes leakage.

**4. Optimization**

Optuna runs 100 trials per model per feature set for the single-site comparison. Each trial evaluates mean RMSE across three time-series CV folds. The best parameters are then retrained on the full training set and evaluated once on the held-out test set.

**5. Feature reduction → Lean Model**

After comparing all six model/feature combinations, the engineered XGBoost variant is analyzed three ways: XGBoost gain importance, SHAP beeswarm, and RFECV with `TimeSeriesSplit`. The 9 features that survive all three are used for the final lean model and carried into the multisite validation.

**6. Multisite validation**

The lean 9-feature pipeline runs independently on each of the 42 sites, separate resampling, separate feature engineering (site-specific lat/lon passed to pvlib), separate Optuna tuning (20 trials), separate 80/20 split. Results are stored in a pickle file and exported as two CSVs: per-site metrics and per-site best hyperparameters.

---

## Feature Engineering

The engineered set is built on top of the baseline in a few distinct layers.

**Solar physics via pvlib**

```python
location = pvlib.location.Location(lat, lon, tz='Australia/Melbourne')
hourly_df['Theoretical_Max_GHI'] = location.get_clearsky(times)['ghi'].values
solar_position = location.get_solarposition(times)
hourly_df['zenith'] = solar_position['zenith'].values
hourly_df['azimuth'] = solar_position['azimuth'].values
```

`Theoretical_Max_GHI` gives the Ineichen clearsky irradiance, what GHI would be on a perfectly clear day at this latitude and time. Pairing that with actual weather tells the model how much the atmosphere is attenuating expected irradiance, which is more informative than weather alone. Zenith and azimuth capture the sun's position directly rather than letting the model infer it from hour-of-day.

**Thermal interactions**

`Temp_Squared` is included as a proxy for panel heat degradation, crystalline silicon panels lose roughly 0.4–0.5% efficiency per °C above their rated temperature, so the relationship between temperature and output isn't linear at high values. `Temp_Humidity_Interact` and `Wind_Temp_Interact` capture how wind cooling modifies the temperature effect.

**7-day lag**

```python
LAG_STEPS = 168
hourly_df['lag_7_days'] = hourly_df['SolarGeneration'].shift(LAG_STEPS)
hourly_df['weekly_trend'] = hourly_df['SolarGeneration'].shift(LAG_STEPS).rolling(window=24, min_periods=1).mean()
```

A 168-step shift gives the actual generation from the same hour one week prior. This is dropped in the lean model after RFECV, but it contributes meaningfully in the full 24-feature set. One consequence: the first 168 rows of each site are discarded, which reduces the usable training window slightly.

**Cyclical time encodings**

Hour and month are encoded as sin/cos pairs rather than raw integers. This avoids the artificial discontinuity at midnight (hour 23 → hour 0) and at the year boundary (December → January) that tree models would otherwise treat as a large numeric step.

**Rolling weather statistics**

24-hour air temperature mean, 6-hour humidity mean, and 3-hour wind speed mean. These represent local atmospheric inertia, temperature doesn't change instantaneously, and a rolling mean gives the model context about the recent trend.

---

## Models and Optimization

Three models are compared: `XGBRegressor`, `LGBMRegressor`, and `RandomForestRegressor`. Optuna searches over a broad hyperparameter space for each, tree depth, learning rate, regularization terms, subsampling rates, and so on. All predictions are clipped to zero (`np.maximum(0, preds)`) since negative generation is physically impossible.

The optimized parameters from the single-site Optuna run are hardcoded into a second training cell for reproducibility. If you re-run the Optuna search, you'll likely get slightly different parameters due to the stochastic nature of the optimizer, results should be close but not identical.

---

## Results

Numbers below are from the single-site experiment (CampusKey=2, SiteKey=1).

| Model | Feature Set | R² | RMSE (kW) | MAE (kW) | NRMSE (%) |
|---|---|---|---|---|---|
| Random Forest | Baseline | 0.7884 | 12.845 | 9.893 | 14.85 |
| LightGBM | Baseline | 0.7865 | 12.900 | 10.275 | 14.91 |
| XGBoost | Baseline | 0.7790 | 13.125 | 10.619 | 15.17 |
| Random Forest | Engineered | 0.8749 | 9.896 | 7.076 | 11.44 |
| LightGBM | Engineered | 0.8711 | 10.044 | 7.325 | 11.61 |
| **XGBoost** | **Engineered** | **0.8817** | **9.623** | **6.920** | **11.13** |

The engineered features reduce RMSE by roughly 3 kW (~25%) across all three models. That's a consistent enough improvement that it's not a fluke of one model's architecture, the physics features are genuinely useful.

### Lean Model

After identifying the best-performing configuration (XGBoost on engineered features), the feature set was reduced from 24 to 9 using a combination of XGBoost gain importance, SHAP analysis, RFECV, and some manual trial and error.

The 9 selected features: `RelativeHumidity`, `WindSpeed`, `Theoretical_Max_GHI`, `zenith`, `Temp_Gradient`, `hour_sin`, `hour_cos`, `ApparentTemperature`, `DewPointTemperature`.

The 7-day lag, rolling weather means, most thermal interaction terms, and azimuth were all dropped. The physics features, clearsky GHI and zenith, survived; the autoregressive ones didn't.

With a different feature set the previous hyperparameters no longer apply, so a fresh 100-trial Optuna study was run on the reduced input. The resulting model is shallower and uses far fewer trees (`n_estimators=141`, `max_depth=4`) than the full engineered XGBoost (`n_estimators=563`, `max_depth=5`), which is expected, less input dimensionality means less capacity is needed. This lean model is what gets carried into the multisite validation across all 42 sites.

---

## Multisite Validation

The lean pipeline runs on each unique `SiteKey` in the dataset (42 total). Each site is processed independently: its own resampling, its own pvlib computation using the site's actual `lat`/`Lon` from `Solar_Site_Details.csv`, its own Optuna study (20 trials), and its own 80/20 chronological split.

Results are saved in two places:

- `all_site_results_optimized.pkl` on Google Drive, contains the trained model objects, best params, and metrics per site
- `Optimized_Hyperparameters_By_Site.csv` and `Final_Test_Results_By_Site.csv`, downloaded locally via `google.colab.files`

The 20-trial Optuna budget for multisite is a practical compromise. The single-site experiment used 100 trials, which takes ~10–15 minutes per model. At 20 trials, the results are good enough to evaluate whether the feature set generalizes.

---

## Known Limitations

**Google Colab dependency.** The notebook is tightly coupled to Colab's Drive mounting and directory structure. Running it locally requires changing the `path` variable and removing the `drive.mount()` call. The `from google.colab import files` export at the end also needs to be replaced.

**Hardcoded timezone.** The pvlib location is initialized with `tz='Australia/Melbourne'` throughout. This is correct for this dataset, but if you adapt this notebook for another region, that needs to change, and the timestamps must be timezone-aware before passing to pvlib (the code uses `tz_localize` to handle this for Colab's default UTC-naive timestamps).

**No cross-site transfer.** Each site trains its own model from scratch. There's no shared representation or transfer between sites. For sites with very short histories, a cross-site ensemble or transfer learning approach could likely perform better.
