#!/usr/bin/env python
# coding: utf-8

# In[1]:

from utils_dir import get_curr_dir, include_home_dir
include_home_dir()

import pandas as pd

from jumpmodels.utils import filter_date_range        # useful helpers
from jumpmodels.jump import JumpModel                 # class of JM & CJM
from jumpmodels.sparse_jump import SparseJumpModel    # class of Sparse JM


# In[2]:


from util.data import dfwind

ticker = "000300.SH"
ticker = "SPX.GI"
cp = dfwind.get_wsddata(ticker,'CLOSE')
cp.index.names = ['date']
cp['ret'] = cp['CLOSE'].pct_change()
cp = cp.dropna()
cp.to_csv(f'{ticker}.csv')


from feature import DataLoader

data = DataLoader(ticker=ticker, ver="v0").load(start_date="2005-01-01", end_date="2024-12-12")

print("Daily returns stored in `data.ret_ser`:", "-"*50, sep="\n")
print(data.ret_ser, "-"*50, sep="\n")
print("Features stored in `data.X`:", "-"*50, sep="\n")
print(data.X)


# ## Train/Test Split and Preprocessing
# 
# We perform a simple time-based split: data from the beginning of 2007 to the end of 2021, covering a 15-year period, is used as the training set for fitting the JMs.
# The period from 2022 to late 2024 is reserved as the test set, where we apply the trained JMs to perform online regime inference.
# We use the helper function `filter_date_range` to filter the start and end dates of a DataFrame.

# In[3]:


train_start, test_start = "2006-1-1", "2014-1-1"
# filter dates
X_train = filter_date_range(data.X, start_date=train_start, end_date=test_start)
X_test = filter_date_range(data.X, start_date=test_start)
# print time split
train_start, train_end = X_train.index[[0, -1]]
test_start, test_end = X_test.index[[0, -1]]
print("Training starts at:", train_start, "and ends at:", train_end)
print("Testing starts at:", test_start, "and ends at:", test_end)

# Preprocessing
from jumpmodels.preprocess import StandardScalerPD, DataClipperStd
clipper = DataClipperStd(mul=3.)
scalar = StandardScalerPD()
# fit on training data
X_train_processed = scalar.fit_transform(clipper.fit_transform(X_train))
# transform the test data
X_test_processed = scalar.transform(clipper.transform(X_test))


# # Original JM

# ## In-Sample Fitting
# 
# We begin by illustrating the in-sample training of the original JM.
# The model parameters are set as follows: the number of components/states/regimes is 2, the jump penalty $\lambda$ is 50.0, and `cont=False`, indicating the original discrete JM that performs hard clustering. 
# It is important to note that the jump penalty $\lambda$ is a crucial hyperparameter that requires tuning, either through statistical criteria or cross-validation (see references for details). 
# 
# The docstring provides comprehensive documentation of all parameters and attributes (thanks to ChatGPT).

# In[5]:


# set the jump penalty
jump_penalty=50.
# initlalize the JM instance
jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )


# In the `.fit()` call, we pass the return series for each period to be used for sorting the states.
# We specify `sort_by="cumret"`, meaning that the state labels (0 or 1) are determined by the cumulative returns under each state. The state with higher cumulative returns is denoted as $s_t=0$ (bull market), and the state with lower returns is denoted as $s_t=1$ (bear market). 
# 

# In[6]:


# call .fit()
jm.fit(X_train_processed, data.ret_ser, sort_by="cumret")


# The cluster centroids for each state are stored in the `centers_` attribute. 
# While these values are scaled, making direct interpretation hard, the bull market state is clearly characterized by higher returns, lower downside deviation, and a higher Sortino ratio, with a distinct separation between the two regimes.

# In[7]:


print("Scaled Cluster Centroids:", pd.DataFrame(jm.centers_, index=["Bull", "Bear"], columns=X_train.columns), sep="\n" + "-"*50 + "\n")


# ### Visualization
# 
# The `jumpmodels.plot` module provides useful functions for visualizing regime identification. 
# We'll use the `labels_` attribute of the JM instance, which contains integers from 0 to `n_c-1`, representing the in-sample fitted regime assignment for each period.
# 
# From the plot, we observe that the identified regimes for the Nasdaq-100 Index successfully capture several significant market downturns, including the global financial crisis, corrections in 2012, 2015-2016, 2019, and the COVID-19 crash in 2020. 
# These identified regimes correspond well to shifts in market fundamentals, as interpreted in hindsight.
# 

# In[8]:


from jumpmodels.plot import plot_regimes_and_cumret, savefig_plt

ax, ax2 = plot_regimes_and_cumret(jm.labels_, data.ret_ser, n_c=2, start_date=train_start, end_date=train_end, )
ax.set(title=f"In-Sample Fitted Regimes by the JM ($\\lambda$={jump_penalty})")
savefig_plt(f"{get_curr_dir()}/plots/JM_lambd-{jump_penalty}_train.pdf")

# ### Modifying Parameters via `set_params()`
# 
# Our model inherits from the `BaseEstimator` class provided by `scikit-learn`, enabling a wide range of utility methods.
# Among these, we highlight the `.set_params()` function, which allows users to reset any input parameters without creating a new instance.
# This functionality is particularly useful when the model needs to be refitted multiple times, such as when testing different jump penalties.
# 
# As an example, we reset the jump penalty to zero, effectively reducing the model to a baseline $k$-means clustering algorithm where temporal information is ignored. 
# This comparison illustrates the value of applying a jump penalty to ensure temporal consistency and reduce the occurrence of unrealistic regime shifts.


# ## Online Inference
# 
# After completing the in-sample training, we apply the trained models for online inference on the test period using the `predict_online()` method. 
# Here, *online inference* means that the regime inference for period $t$ is based solely on the data available up to the end of that period, without using any future data.
# We revert the jump penalty to a reasonable value of 50.0.
# 
# 

# In[11]:

# refit
jump_penalty=50.
jm.set_params(jump_penalty=jump_penalty).fit(X_train_processed, data.ret_ser, sort_by="cumret")
# make online inference 
labels_test_online = jm.predict_online(X_test_processed)

ax, ax2 = plot_regimes_and_cumret(labels_test_online, data.ret_ser, n_c=2, start_date=test_start, end_date=test_end, )
ax.set(title=f"Out-of-Sample Online Inferred Regimes by the JM ($\\lambda$={jump_penalty})")
savefig_plt(f"{get_curr_dir()}/plots/JM_lambd-{jump_penalty}_test_online.pdf")


# In contrast to online inference, the `.predict()` method performs state decoding using all test data (i.e., from 2022 to 2024) at once. 
# While this approach is less realistic for trading applications, we observe that, with access to the full dataset, the model avoids the reversal in late 2024 and exits the bear signal in 2023 slightly earlier than with online inference.
# 
# Though this approach is less applicable for real-world backtesting in financial markets, it holds potential uses in other engineering fields (such as language modeling, where access to an entire sentence is available at once.)

# In[13]:

# make inference using all test data
labels_test = jm.predict(X_test_processed)
# plot
ax, ax2 = plot_regimes_and_cumret(labels_test, data.ret_ser, n_c=2, start_date=test_start, end_date=test_end, )
_ = ax.set(title=f"Out-of-Sample Predicted Regimes by the JM Using All Test Data ($\\lambda$={jump_penalty})")



# # SJM: Sparse JM with Feature Selection
# 
# Finally, the Sparse Jump Model (SJM) introduces feature weighting on top of the original JM or CJM. 
# Features leading to better in-sample clustering effects, as measured by variance reduction, are assigned higher weights, while a LASSO-like constraint on the weight vector ensures that noisy features receive zero weight.
# 
# ## In-Sample Fitting
# 
# ### Parameters
# 
# SJM is implemented in the class `SparseJumpModel`, with an additional parameter `max_feats`, which controls the number of features included.
# This parameter roughly reflects the effective number of features. (In the notation of Nystrup et al. (2021), `max_feats` corresponds to $\kappa^2$.)
# 
# The jump penalty value is of a similar magnitude to the non-sparse model. In this case, we try `max_feats=3.` and `jump_penalty=50.`

# In[17]:


max_feats=3.
jump_penalty=50.
# init sjm instance
sjm = SparseJumpModel(n_components=2, max_feats=max_feats, jump_penalty=jump_penalty, )
# fit
sjm.fit(X_train_processed, ret_ser=data.ret_ser, sort_by="cumret")

# The feature weights are stored in the attribute `feature_weights`. 
# Generally, we observe that features with longer halflives receive higher weights, indicating that less smoothed features are noisier and are excluded from the model, thanks to the feature weighting mechanism.

print("SJM Feature Weights:", "-"*50, sjm.feat_weights, sep="\n")

# plot
ax, ax2 = plot_regimes_and_cumret(sjm.labels_, data.ret_ser, n_c=2, start_date=train_start, end_date=train_end, )
ax.set(title=f"In-Sample Fitted Regimes by the SJM ($\\lambda$={jump_penalty}, $\\kappa^2$={max_feats})")
savefig_plt(f"{get_curr_dir()}/plots/SJM_lambd-{jump_penalty}_max-feats-{max_feats}_train.pdf")


# online inference
labels_test_online_sjm = sjm.predict_online(X_test_processed)

# plot
ax, ax2 = plot_regimes_and_cumret(labels_test_online_sjm, data.ret_ser, start_date=test_start, end_date=test_end, )
ax.set(title=f"Out-of-Sample Online Inferred Regimes by the SJM ($\\lambda$={jump_penalty}, $\\kappa^2$={max_feats})")
savefig_plt(f"{get_curr_dir()}/plots/SJM_lambd-{jump_penalty}_max-feats-{max_feats}_test_online.pdf")





