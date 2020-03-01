# Advanced Pandas Vault
The only Pandas utility package you would ever need. 

1. Table Processing
1. Table Exploration
1. Feature Processing
1. Feature Engineering

### Table Processing

----------

**>>> Memory Reduction Script**

```python

import gc

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        gc.collect()
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
X_train = reduce_mem_usage(X_train)
X_test = reduce_mem_usage(X_test)

```

**>>> Nested Conditional**
```python

list_num = [5,10,15]
nums = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in list_num]
print(nums)
```

### Table Exploration

----------

**>>> Tabled Strong Correlations**

```python
data_corr = data.corr()
# Set the threshold to select only highly correlated attributes
threshold = 0.5
# List of pairs along with correlation above threshold
corr_list = []
#Search for the highly correlated pairs
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index
#Sort to show higher ones first            
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))
#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))
```

**>>> Highly Correlated Pairs**

```python

#Search for the highly correlated pairs
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index

#Sort to show higher ones first            
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

```


**>>> Correlation with Target**

```python

# Correlation amongst top feautres and target
corr = df.corr()
abb = corr["target"].sort_values(ascending=False)[::5].index.values.tolist()
corr = corr[corr.index.isin(abb)]
corr = corr[abb]
corr.ix[:,0].abs().sort_values(ascending=False)
corr.shape

```


**>>> Missing Data**

```python

def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
```


**>>> Shift Columns to Front**

```python

### Bring certain columns to the front:
def ListShuff(items, df):
    cols = list(df)
    for i in range(len(items)):
        cols.insert(i, cols.pop(cols.index(items[i])))
    df = df.ix[:, cols]
    df.reset_index(drop=True, inplace=True)
    return df
```

### Feature Processing

----------

**>>> Drop Mostly Empty Columns**

```python
### Only where prediction is the main priority 
## Rename None to NaN 
df = df.fillna(value=pd.np.nan)
## drop collumns where 95% NaN values
df = df.dropna(thresh = len(df) * (1-0.95),how='all' axis = 1)         ### First Solutions
df = df.dropna(thresh=df.shape[0]*0.05,how='all',axis=1, inplace=True) ### Second Solutions

````

**>>> Drop Constant Column**

```python 
df = df.loc[:,df.apply(pd.Series.nunique) != 1] 
```

**>>> Drop Quasi-Constant Features**

```python

## Removing features that show the same value for the majority/all of the observations 

def constant_feature_detect(data,threshold=0.98):
    """ detect features that show the same value for the 
    majority/all of the observations (constant/quasi-constant features)
    
    Parameters
    ----------
    data : pd.Dataframe
    threshold : threshold to identify the variable as constant
        
    Returns
    -------
    list of variables names
    """
    
    data_copy = data.copy(deep=True)
    quasi_constant_feature = []
    for feature in data_copy.columns:
        predominant = (data_copy[feature].value_counts() / np.float(
                      len(data_copy))).sort_values(ascending=False).values[0]
        if predominant >= threshold:
            quasi_constant_feature.append(feature)
    print(len(quasi_constant_feature),' variables are found to be almost constant')    
    return quasi_constant_feature

# the original dataset has no constant variable
quasi_constant_feature = ft.constant_feature_detect(data=X_train,threshold=0.9)

```

**>>> Distribution Tail Imputation**

```python

## Replace NaN by far distribution:

def impute_NA_with_end_of_distribution(data,NA_col=[]):
    """
    replacing the NA by values that are at the far end of the distribution of that variable
    calculated by mean + 3*std
    """
    
    data_copy = data.copy(deep=True)
    for i in NA_col:
        if data_copy[i].isnull().sum()>0:
            data_copy[i+'_impute_end_of_distri'] = data_copy[i].fillna(data[i].mean()+3*data[i].std())
        else:
            warn("Column %s has no missing" % i)
    return data_copy    
  

data6 = impute_NA_with_end_of_distribution(data=data,NA_col=['Age'])

```


**>>> Outlier Identification Strategies**

```python

def outlier_detect_arbitrary(data,col,upper_fence,lower_fence):
    '''
    identify outliers based on arbitrary boundaries passed to the function.
    '''

    para = (upper_fence, lower_fence)
    tmp = pd.concat([data[col]>upper_fence,data[col]<lower_fence],axis=1)
    outlier_index = tmp.any(axis=1)
    print('Num of outlier detected:',outlier_index.value_counts()[1])
    print('Proportion of outlier detected',outlier_index.value_counts()[1]/len(outlier_index))    
    return outlier_index, para


  
index,para = outlier_detect_arbitrary(data=data,col='Fare',upper_fence=300,lower_fence=5)
print('Upper bound:',para[0],'\nLower bound:',para[1])

```


**>>> Outlier Detection IQR**

```python

def outlier_detect_IQR(data,col,threshold=3):
    '''
    outlier detection by Interquartile Ranges Rule, also known as Tukey's test. 
    calculate the IQR ( 75th quantile - 25th quantile) 
    and the 25th 75th quantile. 
    Any value beyond:
        upper bound = 75th quantile + （IQR * threshold）
        lower bound = 25th quantile - （IQR * threshold）   
    are regarded as outliers. Default threshold is 3.
    '''
     
    IQR = data[col].quantile(0.75) - data[col].quantile(0.25)
    Lower_fence = data[col].quantile(0.25) - (IQR * threshold)
    Upper_fence = data[col].quantile(0.75) + (IQR * threshold)
    para = (Upper_fence, Lower_fence)
    tmp = pd.concat([data[col]>Upper_fence,data[col]<Lower_fence],axis=1)
    outlier_index = tmp.any(axis=1)
    print('Num of outlier detected:',outlier_index.value_counts()[1])
    print('Proportion of outlier detected',outlier_index.value_counts()[1]/len(outlier_index))
    return outlier_index, para

  
index,para = outlier_detect_IQR(data=data,col='Fare',threshold=5)
print('Upper bound:',para[0],'\nLower bound:',para[1])

```

**>>> Outlier Detection Mean Standard Deviation**

```python
def outlier_detect_mean_std(data,col,threshold=3):
    '''
    outlier detection by Mean and Standard Deviation Method.
    If a value is a certain number(called threshold) of standard deviations away 
    from the mean, that data point is identified as an outlier. 
    Default threshold is 3.
    This method can fail to detect outliers because the outliers increase the standard deviation. 
    The more extreme the outlier, the more the standard deviation is affected.
    '''
   
    Upper_fence = data[col].mean() + threshold * data[col].std()
    Lower_fence = data[col].mean() - threshold * data[col].std()   
    para = (Upper_fence, Lower_fence)   
    tmp = pd.concat([data[col]>Upper_fence,data[col]<Lower_fence],axis=1)
    outlier_index = tmp.any(axis=1)
    print('Num of outlier detected:',outlier_index.value_counts()[1])
    print('Proportion of outlier detected',outlier_index.value_counts()[1]/len(outlier_index))
    return outlier_index, para
  
index,para = ot.outlier_detect_mean_std(data=data,col='Fare',threshold=3)
print('Upper bound:',para[0],'\nLower bound:',para[1])  
  
# too aggressive for our dataset, about 18% of cases are detected as outliers.
index = outlier_detect_MAD(data=data,col='Fare',threshold=3.5)

```

**>>> Outlier Detection MAD**

```python

def outlier_detect_MAD(data,col,threshold=3.5):
    """
    outlier detection by Median and Median Absolute Deviation Method (MAD)
    The median of the residuals is calculated. Then, the difference is calculated between each historical value and this median. 
    These differences are expressed as their absolute values, and a new median is calculated and multiplied by 
    an empirically derived constant to yield the median absolute deviation (MAD). 
    If a value is a certain number of MAD away from the median of the residuals, 
    that value is classified as an outlier. The default threshold is 3 MAD.
    
    This method is generally more effective than the mean and standard deviation method for detecting outliers, 
    but it can be too aggressive in classifying values that are not really extremely different. 
    Also, if more than 50% of the data points have the same value, MAD is computed to be 0, 
    so any value different from the residual median is classified as an outlier.
    """
    
    median = data[col].median()
    median_absolute_deviation = np.median([np.abs(y - median) for y in data[col]])
    modified_z_scores = pd.Series([0.6745 * (y - median) / median_absolute_deviation for y in data[col]])
    outlier_index = np.abs(modified_z_scores) > threshold
    print('Num of outlier detected:',outlier_index.value_counts()[1])
    print('Proportion of outlier detected',outlier_index.value_counts()[1]/len(outlier_index))
    return outlier_index

  # too aggressive for our dataset, about 18% of cases are detected as outliers.
index = ot.outlier_detect_MAD(data=data,col='Fare',threshold=3.5)
```

**>>> Outlier Imputation Strategies**

```python
def impute_outlier_with_arbitrary(data,outlier_index,value,col=[]):
    """
    impute outliers with arbitrary value
    """
    
    data_copy = data.copy(deep=True)
    for i in col:
        data_copy.loc[outlier_index,i] = value
    return data_copy
    
# use any of the detection method above
index,para = ot.outlier_detect_arbitrary(data=data,col='Fare',upper_fence=300,lower_fence=5)
print('Upper bound:',para[0],'\nLower bound:',para[1])


print(data2[255:275])

# see index 258,263,271 have been replaced
data2 = ot.impute_outlier_with_arbitrary(data=data,outlier_index=index,
                                         value=-999,col=['Fare'])
print(data2[255:275])

```

**>>> Winzorization**

```python

## Clips instead of drops the data
    
def windsorization(data,col,para,strategy='both'):
    """
    top-coding & bottom coding (capping the maximum of a distribution at an arbitrarily set value,vice versa)
    """
    
    data_copy = data.copy(deep=True)  
    if strategy == 'both':
        data_copy.loc[data_copy[col]>para[0],col] = para[0]
        data_copy.loc[data_copy[col]<para[1],col] = para[1]
    elif strategy == 'top':
        data_copy.loc[data_copy[col]>para[0],col] = para[0]
    elif strategy == 'bottom':
        data_copy.loc[data_copy[col]<para[1],col] = para[1]  
    return data_copy

  # use any of the detection method above
index,para = ot.outlier_detect_arbitrary(data,'Fare',300,5)
print('Upper bound:',para[0],'\nLower bound:',para[1])


# see index 258,263,271 have been replaced with top/bottom coding

data3 = ot.windsorization(data=data,col='Fare',para=para,strategy='both')
data3[255:275]

```

**>>> Drop Outliers**

```python

## Drop
def drop_outlier(data,outlier_index):
    """
    drop the cases that are outliers
    """
    data_copy = data[~outlier_index]
    return data_copy
# drop the outlier.
# we can see no more observations have value >300 or <5. They've been removed.
data4 = ot.drop_outlier(data=data,outlier_index=index)

```

**>>> Impute Outlier with Average**

```python

def impute_outlier_with_avg(data,col,outlier_index,strategy='mean'):
    """
    impute outlier with mean/median/most frequent values of that variable.
    """
    
    data_copy = data.copy(deep=True)
    if strategy=='mean':
        data_copy.loc[outlier_index,col] = data_copy[col].mean()
    elif strategy=='median':
        data_copy.loc[outlier_index,col] = data_copy[col].median()
    elif strategy=='mode':
        data_copy.loc[outlier_index,col] = data_copy[col].mode()[0]   
        
    return data_copy
  
data5 = ot.impute_outlier_with_avg(data=data,col='Fare',
                                 outlier_index=index,strategy='mean')
                                 
```                               



### Feature Engineering

----------


**>>> Normalisation**

```python
# Normalising a dataframe Normalise
  def Normalisation(low,high,df):
      listed = list(df)
      scaler = MinMaxScaler(feature_range=(low,high))
      scaled = scaler.fit_transform(df)
      df = pd.DataFrame(scaled)
      df.columns = listed
      return df
# Standardising dataframe
```

**>>> Standardisation**

```python
from sklearn.preprocessing import StandardScaler

def Standardisation(df):
    listed = list(df)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    df = pd.DataFrame(scaled)
    df.columns = listed
    return df
```


**>>> Scaler**

```python

def scaler(df,scaler=None,train=True, target=None):
  if target:
    x = df.drop([target],axis=1).values #returns a numpy array
  else:
    x = df.values
  if train:
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(x)
    x_scaled = min_max_scaler.transform(x)
  else:
    x_scaled = scaler.transform(x)
  
  if target:
    df_out = pd.DataFrame(x_scaled, index=df.index, columns=df.drop([target],axis=1).columns)
    df_out[target]= df[target]
  else:
    df_out = pd.DataFrame(x_scaled, index=df.index, columns=df.columns)
  if train:
    return df_out, min_max_scaler
  else:
    return df_out
    
```

**>>> Automate Dummy (one-hot) Encoding**

```python
# Creating dummies for small object uniques
list_dummies =[]
for col in df.columns:
    if (len(df[col].unique()) <15):
        list_dummies.append(col)
        print(col)
df_edit = pd.get_dummies(df, columns = list_dummies) # Saves original dataframe
df_edit = pd.concat([df[["year","qtr"]],df_edit],axis=1)
```

**>>> Binarise Empty Columns**

```python
# Binarise slightly empty columns
df = df_edit.copy()
this =[]
for col in df.columns:
    if df[col].dtype != "object":
        is_null = df[col].isnull().astype(int).sum()
        if (is_null/df.shape[0]) >0.70: # if more than 70% is null binarise
            print(col)
            this.append(col)
            df[col] = df[col].astype(float)
            df[col] = df[col].apply(lambda x: 0 if (np.isnan(x)) else 1)
df = pd.get_dummies(df, columns = this) 
```

**>>> Target Encoding**

```python

# taken from https://medium.com/@pouryaayria/k-fold-target-encoding-dfe9a594874b
from sklearn import base
from sklearn.model_selection import KFold

class KFoldTargetEncoderTrain(base.BaseEstimator,
                               base.TransformerMixin):
    def __init__(self,colnames,targetName,
                  n_fold=5, verbosity=True,
                  discardOriginal_col=False):
        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        assert(type(self.targetName) == str)
        assert(type(self.colnames) == str)
        assert(self.colnames in X.columns)
        assert(self.targetName in X.columns)
        mean_of_target = X[self.targetName].mean()
        kf = KFold(n_splits = self.n_fold,
                   shuffle = True, random_state=2019)
        col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
        X[col_mean_name] = np.nan
        for tr_ind, val_ind in kf.split(X):
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(X_tr.groupby(self.colnames)
                                     [self.targetName].mean())
            X[col_mean_name].fillna(mean_of_target, inplace = True)
        if self.verbosity:
            encoded_feature = X[col_mean_name].values
            print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,self.targetName,                    
                   np.corrcoef(X[self.targetName].values,
                               encoded_feature)[0][1]))
        if self.discardOriginal_col:
            X = X.drop(self.targetName, axis=1)
        return X
      
      
targetc = KFoldTargetEncoderTrain('Pclass','Survived',n_fold=5)
new_train = targetc.fit_transform(train)

new_train[['Pclass_Kfold_Target_Enc','Pclass']]

```

**>>> Polynomials**

```python

def Polynomials(df):
    x1 = df['Feature_1'].fillna(-1) + 1e-1
    x2 = df['Feature_2'].fillna(-1) + 1e-1
    x3 = df['Feature_3'].fillna(-1) + 1e-1
    
    df['Feature_1/Feature_2'] = x1/x2
    df['Feature_2/Feature_1'] = x2/x1
    df['Feature_1/Feature_3'] = x1/x3
    df['Feature_3/Feature_1'] = x3/x1
    df['Feature_2/Feature_3'] = x2/x3
    df['Feature_3/Feature_2'] = x3/x2
    
    return df
    
```

**>>> Genetic Programming**

```python

import gplearn 
function_set = ['add', 'sub', 'mul', 'div',
                'sqrt', 'log', 'abs', 'neg', 'inv','tan']

gp = SymbolicTransformer(generations=800, population_size=200,
                         hall_of_fame=100, n_components=10,
                         function_set=function_set,
                         parsimony_coefficient=0.0005,
                         max_samples=0.9, verbose=1,
                         random_state=0, n_jobs=6)

gp.fit(train_df.drop("TARGET", axis=1), train_df["TARGET"])
gp_features.to_csv("gp_features.csv",index=False)

```

**>>> Shap Engineering**

```python

import shap
## Model should be LightGBM or XGBoost
shap_values = shap.TreeExplainer(model).shap_values(X_train)

shap_fram = pd.DataFrame(shap_values[:,:-1], columns=list(X_train.columns))

shap_new = shap_fram.sum().sort_values().to_frame()

shap_new.columns = ["SHAP"]

low = shap_new[shap_new["SHAP"]<shap_new.quantile(.20).values[0]].reset_index()

high = shap_new[shap_new["SHAP"]>shap_new.quantile(.80).values[0]].reset_index()

## Copy and pasted the output to these eliments 
low = ['i7', 'i8', 'i9', 'i10', 'i11']

high = ['i3', 'i15', 'i12', 'i35', 'EXT_SOURCE_3']


for h, l in zip(high, low):
    playdata[h+"_"+l] = playdata[l]/playdata[h] ### You can create other interactions.
    playdata[h+"_"+l] = playdata[h+"_"+l].replace([np.inf, -np.inf], np.nan)
    playdata[h+"_"+l] = playdata[h+"_"+l].fillna(value=0)
    
```

**>>> PCA Features**

```python

import seaborn as sns
from sklearn.decomposition import PCA
pca2 = PCA(n_components=2)
pca2_results = pca2.fit_transform(df_2.drop(["TARGET"], axis=1))

for i in range(pca2_results.shape[1]):
    df_2["pca_"+str(i)] = pca2_results[:,i]
    
```

**>>> Date Features**

```python

# Additional Date Features
df_edit = df_gvkey
df_edit["public_date"] = pd.to_datetime(df_edit["public_date"])
df_edit["public_date_month"] = df_edit["public_date"].dt.month.astype(int)
df_edit["public_date_year"]  = df_edit["public_date"].dt.year.astype(int)
df_edit["public_date_week"]  = df_edit["public_date"].dt.week.astype(int)
df_edit["public_date_day"]   = df_edit["public_date"].dt.day.astype(int)
df_edit["public_date_dayofweek"]= df_edit["public_date"].dt.dayofweek.astype(int)
df_edit["public_date_dayofyear"]= df_edit["public_date"].dt.dayofyear.astype(int)
df_edit["public_date_hour"] = df_edit["public_date"].dt.hour.astype(int)
df_edit["public_date_int"] = pd.to_datetime(df_edit["public_date"]).astype(int)

```










