import numpy as np
import pandas as pd
import os

labelpath = 'labels\\part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv'  # specify path to label file
csv_dir = 'features'    # specify folder for features csv file

# create empty dataframe to store features and label
features_df = pd.DataFrame()
label_df = pd.DataFrame()

# read features files
for csv_file in os.listdir(csv_dir):
    if csv_file == '.DS_Store':
        continue
    filepath = csv_dir+'\\'+csv_file
    features_df = pd.concat([features_df, pd.read_csv(filepath)])

# read label files
label_df = pd.read_csv(labelpath)

# calculate total number of booking and label
num_booking = features_df['bookingID'].nunique()
num_label = label_df['bookingID'].nunique()
print(num_booking)
print(num_label)

# get minimum number of timestep for each booking
max_second = list()
for ID in features_df['bookingID'].unique():
    values = list()
    df = features_df[features_df['bookingID']==ID]
    for value in df['second'].unique():
        values.append(value)
    max_second.append(max(values))
num_timestep = int(min(max_second))
print(num_timestep)


num_timestep = 119  # Set the number of timestep to 119 since the lowest timestep in the given data is 119
m = num_booking    # number of samples
n = 9   # number of fetures

# convert features to numpy array
i=0
num_neg = 0
x = np.empty([m,num_timestep,n])
y = np.empty([m,1])
for ID in features_df['bookingID'].unique():
    df = label_df[label_df['bookingID']==ID]
    label = int(df['label'].values[0])
    # # Down sampling negative eaxamples to balance datasets
    # if label == 0:
    #     num_neg = num_neg+1
    #     if num_neg > 4990:
    #         continue 
    y[i,:] = label
    df = features_df[features_df['bookingID']==ID]
    del df['bookingID']
    del df['Accuracy']
    x[i,:,:] = df.values[0:num_timestep,:]
    # i=i+1
    # if i > m-1:
    #     break

print(x.shape)
print(y.shape)
np.save('data\\input.npy', x)
np.save('data\\output.npy', y)

