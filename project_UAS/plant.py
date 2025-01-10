#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# In[2]:


# Set a random seed for reproducibility
np.random.seed(42)

# Define the number of samples
k_max = 10000
t = np.linspace(0, k_max, k_max)  # Time variable

# Generate random inputs
u = 2 * np.random.uniform(-5, 5, k_max)

# Initialize y(k) array
y = np.zeros(k_max)

# Define y(k) calculations
for k in range(1, k_max):
    y[k] = 1 / (1 + (y[k-1])**2) + 0.25 * u[k] - 0.3 * u[k-1]

# Shift arrays for y(k-1), y(k-2), u(k-1), u(k-2)
y_k_1 = np.zeros(k_max)
y_k_2 = np.zeros(k_max)
u_k_1 = np.zeros(k_max)
u_k_2 = np.zeros(k_max)

y_k_1[1:] = y[:-1]
y_k_2[2:] = y[:-2]
u_k_1[1:] = u[:-1]
u_k_2[2:] = u[:-2]


# In[6]:


t.shape, u.shape, y.shape, y_k_1.shape, y_k_2.shape, u_k_1.shape, u_k_2.shape


# In[7]:


# Set up Seaborn style
sns.set_theme(style="whitegrid")

# Create a single figure with a 3x2 grid of subplots
fig, ax = plt.subplots(3, 2, figsize=(15, 15), sharex=True)

# Plot u(k) over time
ax[0, 0].plot(t, u, label='u(k)', color='c')
ax[0, 0].set_ylabel('u(k)')
ax[0, 0].set_title('Plot of u(k) over Time')
ax[0, 0].legend()
ax[0, 0].grid(True)

# Plot y(k) over time
ax[0, 1].plot(t, y, label='y(k)', color='b')
ax[0, 1].set_ylabel('y(k)')
ax[0, 1].set_title('Plot of y(k) over Time')
ax[0, 1].legend()
ax[0, 1].grid(True)

# Plot u(k-1) over time
ax[1, 0].plot(t, u_k_1, label='u(k-1)', color='m')
ax[1, 0].set_ylabel('u(k-1)')
ax[1, 0].set_title('Plot of u(k-1) over Time')
ax[1, 0].legend()
ax[1, 0].grid(True)

# Plot y(k-1) over time
ax[1, 1].plot(t, y_k_1, label='y(k-1)', color='r')
ax[1, 1].set_ylabel('y(k-1)')
ax[1, 1].set_title('Plot of y(k-1) over Time')
ax[1, 1].legend()
ax[1, 1].grid(True)

# Plot u(k-2) over time
ax[2, 0].plot(t, u_k_2, label='u(k-2)', color='y')
ax[2, 0].set_xlabel('Time')
ax[2, 0].set_ylabel('u(k-2)')
ax[2, 0].set_title('Plot of u(k-2) over Time')
ax[2, 0].legend()
ax[2, 0].grid(True)

# Plot y(k-2) over time
ax[2, 1].plot(t, y_k_2, label='y(k-2)', color='g')
ax[2, 1].set_ylabel('y(k-2)')
ax[2, 1].set_title('Plot of y(k-2) over Time')
ax[2, 1].legend()
ax[2, 1].grid(True)

# Show the combined plot
plt.tight_layout()
plt.show()


# In[26]:


X = np.column_stack((u, u_k_1, u_k_2, y_k_1, y_k_2))
y = y.reshape(-1, 1)
u = u.reshape(-1, 1)
u_k_1 = u_k_1.reshape(-1, 1)
u_k_2 = u_k_2.reshape(-1, 1)
y_k_1 = y_k_1.reshape(-1, 1)
y_k_2 = y_k_2.reshape(-1, 1)


# In[27]:


input_scaler = MinMaxScaler(feature_range=(-1, 1))
input_scaler.fit(u.reshape(-1, 1))
output_scaler = MinMaxScaler(feature_range=(-1, 1))
output_scaler.fit(y.reshape(-1, 1))


# In[28]:


u_norm = input_scaler.transform(u.reshape(-1, 1)).flatten()
y_norm = output_scaler.transform(y.reshape(-1, 1)).flatten()
u_k_1_norm = input_scaler.transform(u_k_1.reshape(-1, 1)).flatten()
u_k_2_norm = input_scaler.transform(u_k_2.reshape(-1, 1)).flatten()
y_k_1_norm = output_scaler.transform(y_k_1.reshape(-1, 1)).flatten()
y_k_2_norm = output_scaler.transform(y_k_2.reshape(-1, 1)).flatten()


# In[29]:


# Define the arrays for each variable
X_norm = np.column_stack((u_norm, u_k_1_norm, u_k_2_norm, y_k_1_norm, y_k_2_norm))
y_norm = y_norm.reshape(-1, 1)
u_norm = u_norm.reshape(-1, 1)
u_k_1_norm = u_k_1_norm.reshape(-1, 1)
u_k_2_norm = u_k_2_norm.reshape(-1, 1)
y_k_1_norm = y_k_1_norm.reshape(-1, 1)
y_k_2_norm = y_k_2_norm.reshape(-1, 1)


# In[30]:


X.shape, y.shape, u.shape, u_k_1.shape, u_k_2.shape, y_k_1.shape, y_k_2.shape


# In[31]:


X_df = pd.DataFrame(X, columns=['u', 'u(k-1)', 'u(k-2)', 'y(k-1)', 'y(k-2)'])
y_df = pd.DataFrame(y, columns=['y'])
X_norm_df = pd.DataFrame(X_norm, columns=['u', 'u(k-1)', 'u(k-2)', 'y(k-1)', 'y(k-2)'])
y_norm_df = pd.DataFrame(y_norm, columns=['y'])
df = pd.concat([X_df, y_df], axis=1)
norm_df = pd.concat([X_norm_df, y_norm_df], axis=1)


# In[34]:


df.describe()


# In[35]:


norm_df.describe()


# In[40]:


# Calculate the split index
split_index = int(0.8 * len(X_norm))

# Split the data
X_norm_train = X_norm[:split_index]
X_norm_val = X_norm[split_index:]
y_norm_train = y_norm[:split_index]
y_norm_val = y_norm[split_index:]


# In[41]:


X_norm_train.shape, X_norm_val.shape, y_norm_train.shape, y_norm_val.shape


# In[ ]:




