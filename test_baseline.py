import torch
import random
import numpy as np
import xarray as xr
import pickle as pkl
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torchmetrics import AUROC, Accuracy, AveragePrecision, F1Score
from tqdm import tqdm

number_of_train_years = 16
days_per_week = 8
timeseries_weeks = 48
aggregation_in_weeks = 4  # aggregate per month
year_in_weeks = 48
max_week_with_data = 918
target_count = 1
target_length = 4

# Europe
min_lon = -25
min_lat = 36
max_lon = 50
max_lat = 72

cube_path = "../1d_SeasFireCube.zarr"
cube = xr.open_zarr(cube_path, consolidated=False)

time_train = (
    timeseries_weeks,
    year_in_weeks * number_of_train_years
    - (target_count * target_length),
)
time_val = (
    year_in_weeks * number_of_train_years + timeseries_weeks,
    year_in_weeks * number_of_train_years
    + 2 * timeseries_weeks,
)
time_test = (
    year_in_weeks * number_of_train_years
    + 2 * timeseries_weeks,
    max_week_with_data - (target_count * target_length),
)
start_time, end_time = time_test
print(time_test)
end_time = 894

sample_region = cube.sel(
    latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon)
).isel(time=slice(start_time, end_time))
sample_region.load()

###################### mean seasonal cycle per month ########################
sample_region_group = sample_region.gwis_ba.groupby("time.month").mean()
sample_region_group.load()

print(sample_region_group)
# sample_region_gwsi_ba_values = sample_region.gwis_ba.values

###################### True values ##########################
sample_region_gwsi_ba_values = sample_region.gwis_ba.values
sample_region_gwsi_non_nan = sample_region_gwsi_ba_values >= 0.0

sample_len_non_nan =   np.sum(sample_region_gwsi_non_nan)

samples = []
all_samples_index = np.argwhere(sample_region_gwsi_non_nan)

for index in all_samples_index:
    samples.append(
        (
            sample_region.latitude.values[index[1]],
            sample_region.longitude.values[index[2]],
            sample_region.time.values[index[0]],
        ),
    )
print("Samples len:", len(samples))

first_sample_index = 0
ground_truth = []
ground_truth_mean = []

for idx in tqdm(range(0, len(samples))):
    if idx < first_sample_index:
        continue
    if os.path.exists(os.path.join('data3/test', "graph_{}.pt".format(idx))):
        # logger.info("Skipping sample {} generation.".format(idx))
        continue
    center_lat, center_lon, center_time = samples[idx]

    start_time = center_time
    end_time = center_time + np.timedelta64(
        (target_count * target_length) * days_per_week, "D"
    )

    end_str = str(end_time)
    
    marker1 = end_str.find('-') + 1
    marker2 = end_str.find('T', marker1)
    substr = end_str[marker1:marker2]

    month = int((substr.split('-'))[0])
    day = (substr.split('-'))[1]
    
    
    if int(day)<15:
        if month == 1:
            month = 12
        elif month == 10:
            month = 10
        else:
            month = month - 1

    month = month + target_count
    if month > 12:
        month = month - 12
    if month == 7:
        month = 6
    elif month == 8:
        month = 6
    elif month == 9:
        month = 10
    
    val = sample_region_group.sel(
        latitude=center_lat,
        longitude=center_lon,
        month=month,
    ).values

    if val > 0.0:
        ground_truth_mean.append(1.0)
    else:
        ground_truth_mean.append(0.0)

    # Computing ground truth for lat=", center_lat," lon=", center_lon," time=[", start_time,",", end_time,"]
    
    values = cube["gwis_ba"].sel(
        latitude=center_lat,
        longitude=center_lon,
        time=slice(
            start_time,
            end_time,
        ),
    )

    aggregation_in_days = "{}D".format(target_length * days_per_week)

    aggregated_values = values.resample(
        time=aggregation_in_days, closed="left"
    ).sum(skipna=True)

    ground_truth.append(aggregated_values.values[: target_count])

###################### SAVING GROUND TRUTHS ###############################    

ground_truth_mean = np.array(ground_truth_mean)
ground_truth_mean = torch.from_numpy(ground_truth_mean)

torch.save(ground_truth_mean, 'ground_truth_mean.pt')

ground_truth = np.array(ground_truth)
ground_truth = torch.from_numpy(ground_truth)

torch.save(ground_truth, 'ground_truth.pt')

###################### LOADING AND PRINTING RESULTS ###############################

mean_truth = torch.load("ground_truth_mean.pt")

truth = torch.load("ground_truth.pt")
truth = truth[:, [0]]

for i in range(0, mean_truth.shape[0]):
    mean_truth[i] = torch.where(mean_truth[i] > 0.0, 1, 0)

for i in range(0, truth.shape[0]):
    truth[i] = torch.where(truth[i] > 0.0, 1, 0)
truth = truth.squeeze(1)

acc = Accuracy(task="binary")
prec = acc(mean_truth, truth)
print("Accuracy: ", prec)

average_precision = AveragePrecision(task="binary")
prec = average_precision(mean_truth, truth)
print("AveragePrecision: ", prec)

f1 = F1Score(task="binary", average="macro")
prec = f1(mean_truth, truth)
print("F1Score: ", prec)

auroc = AUROC(task="binary")
prec = auroc(mean_truth, truth)
print("AUROC: ", prec)