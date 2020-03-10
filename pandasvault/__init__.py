import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from math import sin, cos, sqrt, atan2, radians


### TABLE PROCESSING
#=============================================================================================

def pd_config():
    options = {
        'display': {
            'max_colwidth': 25,
            'expand_frame_repr': False,  # Don't wrap to multiple pages
            'max_rows': 14,
            'max_seq_items': 50,         # Max length of printed sequence
            'precision': 4,
            'show_dimensions': False
        },
        'mode': {
            'chained_assignment': None   # Controls SettingWithCopyWarning
        }
    }

    for category, option in options.items():
        for op, value in option.items():
            pd.set_option(f'{category}.{op}', value)  # Python 3.6+

if __name__ == '__main__':
    pd_config()


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

def list_shuff(items, df):
    "Bring a list of columns to the front"
    cols = list(df)
    for i in range(len(items)):
        cols.insert(i, cols.pop(cols.index(items[i])))
    df = df.loc[:, cols]
    df.reset_index(drop=True, inplace=True)
    return df

### TABLE EXPLORATION
#=============================================================================================


def corr_list(df):

  return  (df.corr()
          .unstack()
          .sort_values(kind="quicksort",ascending=False)
          .drop_duplicates().iloc[1:]); df_out
          

def missing_data(data):
    "Create a dataframe with a percentage and count of missing values"
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



### FEATURE PROCESSING
#=============================================================================================

def drop_corr(df, thresh=0.99,keep_cols=[]):
    df_corr = df.corr().abs()
    upper = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(np.bool))
    to_remove = [column for column in upper.columns if any(upper[column] > thresh)] ## Change to 99% for selection
    to_remove = [x for x in to_remove if x not in keep_cols]
    df_corr = df_corr.drop(columns = to_remove)
    return df.drop(to_remove,axis=1)



def replace_small_cat(df, columns, thresh=0.2, term="other"):
  for col in columns:

    # Step 1: count the frequencies
    frequencies = df[col].value_counts(normalize = True)

  # Step 2: establish your threshold and filter the smaller categories

    small_categories = frequencies[frequencies < thresh].index

    df[col] = df[col].replace(small_categories, "Other")
    
  return df



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


def scaler(df,scaler=None,train=True, target=None, cols_ignore=None, type="Standard"):

  if cols_ignore:
    hold = df[cols_ignore].copy()
    df = df.drop(cols_ignore,axis=1)
  if target:
    x = df.drop([target],axis=1).values #returns a numpy array
  else:
    x = df.values
  if train:
    if type=="Standard":
      scal = StandardScaler()
    elif type=="MinMax":
      scal = MinMaxScaler()
    scal.fit(x)
    x_scaled = scal.transform(x)
  else:
    x_scaled = scaler.transform(x)
  
  if target:
    df_out = pd.DataFrame(x_scaled, index=df.index, columns=df.drop([target],axis=1).columns)
    df_out[target]= df[target]
  else:
    df_out = pd.DataFrame(x_scaled, index=df.index, columns=df.columns)
  
  df_out = pd.concat((hold,df_out),axis=1)
  if train:
    return df_out, scal
  else:
    return df_out

def impute_null_with_tail(df,cols=[]):
    """
    replacing the NA by values that are at the far end of the distribution of that variable
    calculated by mean + 3*std
    """
    
    df = df.copy(deep=True)
    for i in cols:
        if df[i].isnull().sum()>0:
            df[i] = df[i].fillna(df[i].mean()+3*df[i].std())
        else:
            warn("Column %s has no missing" % i)
    return df    
  

def outlier_detect(data,col,threshold=3,method="IQR"):
  
    if method == "IQR":
      IQR = data[col].quantile(0.75) - data[col].quantile(0.25)
      Lower_fence = data[col].quantile(0.25) - (IQR * threshold)
      Upper_fence = data[col].quantile(0.75) + (IQR * threshold)
    if method == "STD":
      Upper_fence = data[col].mean() + threshold * data[col].std()
      Lower_fence = data[col].mean() - threshold * data[col].std()   
    if method == "OWN":
      Upper_fence = data[col].mean() + threshold * data[col].std()
      Lower_fence = data[col].mean() - threshold * data[col].std() 
    if method =="MAD":
      median = data[col].median()
      median_absolute_deviation = np.median([np.abs(y - median) for y in data[col]])
      modified_z_scores = pd.Series([0.6745 * (y - median) / median_absolute_deviation for y in data[col]])
      outlier_index = np.abs(modified_z_scores) > threshold
      print('Num of outlier detected:',outlier_index.value_counts()[1])
      print('Proportion of outlier detected',outlier_index.value_counts()[1]/len(outlier_index))
      return outlier_index, (median_absolute_deviation, median_absolute_deviation)


    para = (Upper_fence, Lower_fence)
    tmp = pd.concat([data[col]>Upper_fence,data[col]<Lower_fence],axis=1)
    outlier_index = tmp.any(axis=1)
    print('Num of outlier detected:',outlier_index.value_counts()[1])
    print('Proportion of outlier detected',outlier_index.value_counts()[1]/len(outlier_index))
    
    return outlier_index, para
    

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


def impute_outlier(data,col,outlier_index,strategy='mean'):
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
  

### FEATURE ENGINEERING
#=============================================================================================

def auto_dummy(df, unique=15):
  # Creating dummies for small object uniques
  if len(df)<unique:
    raise ValueError('unique is set higher than data lenght')
  list_dummies =[]
  for col in df.columns:
      if (len(df[col].unique()) < unique):
          list_dummies.append(col)
          print(col)
  df_edit = pd.get_dummies(df, columns = list_dummies) # Saves original dataframe
  #df_edit = pd.concat([df[["year","qtr"]],df_edit],axis=1)
  return df_edit


def binarise_empty(df, frac=80):
  # Binarise slightly empty columns
  this =[]
  for col in df.columns:
      if df[col].dtype != "object":
          is_null = df[col].isnull().astype(int).sum()
          if (is_null/df.shape[0]) >frac: # if more than 70% is null binarise
              print(col)
              this.append(col)
              df[col] = df[col].astype(float)
              df[col] = df[col].apply(lambda x: 0 if (np.isnan(x)) else 1)
  df = pd.get_dummies(df, columns = this) 
  return df


def polynomials(df, feature_list):
  for feat in feature_list:
    for feat_two in feature_list:
      if feat==feat_two:
        continue
      else:
       df[feat+"/"+feat_two] = df[feat]/(df[feat_two]-df[feat_two].min()) #zero division guard
       df[feat+"X"+feat_two] = df[feat]*(df[feat_two])

  return df


def transformations(df,features):
  df_new = df[features]
  df_new = df_new - df_new.min()

  sqr_name = [str(fa)+"_POWER_2" for fa in df_new.columns]
  log_p_name = [str(fa)+"_LOG_p_one_abs" for fa in df_new.columns]
  rec_p_name = [str(fa)+"_RECIP_p_one" for fa in df_new.columns]
  sqrt_name = [str(fa)+"_SQRT_p_one" for fa in df_new.columns]

  df_sqr = pd.DataFrame(np.power(df_new.values, 2),columns=sqr_name, index=df.index)
  df_log = pd.DataFrame(np.log(df_new.add(1).abs().values),columns=log_p_name, index=df.index)
  df_rec = pd.DataFrame(np.reciprocal(df_new.add(1).values),columns=rec_p_name, index=df.index)
  df_sqrt = pd.DataFrame(np.sqrt(df_new.abs().add(1).values),columns=sqrt_name, index=df.index)

  dfs = [df, df_sqr, df_log, df_rec, df_sqrt]

  df=  pd.concat(dfs, axis=1)

  return df


def pca_feature(df, memory_issues=False,mem_iss_component=False,variance_or_components=0.80,drop_cols=None):

  if memory_issues:
    if not mem_iss_component:
      raise ValueError("If you have memory issues, you have to preselect mem_iss_component")
    pca = IncrementalPCA(mem_iss_component)
  else:
    if variance_or_components>1:
      pca = PCA(n_components=variance_or_components) 
    else: # automted selection based on variance
      pca = PCA(n_components=variance_or_components,svd_solver="full") 
  X_pca = pca.fit_transform(df.drop(drop_cols,axis=1))
  df = pd.concat((df[drop_cols],pd.DataFrame(X_pca, columns=["PCA_"+str(i+1) for i in range(X_pca.shape[1])])),axis=1)
  return df


def multiple_lags(df, start=1, end=3,columns=None):
  if not columns:
    columns = df.columns.to_list()
  lags = range(start, end+1)  # Just two lags for demonstration.

  df = df.assign(**{
      '{}_t_{}'.format(col, t): df[col].shift(t)
      for t in lags
      for col in columns
  })
  return df


def multiple_rolling(df, windows = [1,2], functions=["mean","std"], columns=None):
  windows = [1+a for a in windows]
  if not columns:
    columns = df.columns.to_list()
  rolling_dfs = (df[columns].rolling(i)                                    # 1. Create window
                  .agg(functions)                                # 1. Aggregate
                  .rename({col: '{0}_{1:d}'.format(col, i)
                                for col in columns}, axis=1)  # 2. Rename columns
                for i in windows)                                # For each window
  df_out = pd.concat((df, *rolling_dfs), axis=1)
  da = df_out.iloc[:,len(df.columns):]
  da = [col[0] + "_" + col[1] for col in  da.columns.to_list()]
  df_out.columns = df.columns.to_list() + da 

  return  df_out                      # 3. Concatenate dataframes


def date_features(df, date="date"):
  df[date] = pd.to_datetime(df[date])
  df[date+"_month"] = df[date].dt.month.astype(int)
  df[date+"_year"]  = df[date].dt.year.astype(int)
  df[date+"_week"]  = df[date].dt.week.astype(int)
  df[date+"_day"]   = df[date].dt.day.astype(int)
  df[date+"_dayofweek"]= df[date].dt.dayofweek.astype(int)
  df[date+"_dayofyear"]= df[date].dt.dayofyear.astype(int)
  df[date+"_hour"] = df[date].dt.hour.astype(int)
  df[date+"_int"] = pd.to_datetime(df[date]).astype(int)
  return df

def haversine_distance(row, lon="latitude", lat="longitude"):
    c_lat,c_long = radians(52.5200), radians(13.4050)
    R = 6373.0
    long = radians(row['longitude'])
    lat = radians(row['latitude'])
    
    dlon = long - c_long
    dlat = lat - c_lat
    a = sin(dlat / 2)**2 + cos(lat) * cos(c_lat) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c

### MODEL FALIDATION
#=============================================================================================

def classification_scores(y_test, y_predict, y_prob):

  confusion_mat = confusion_matrix(y_test,y_predict)

  TN = confusion_mat[0][0]
  FP = confusion_mat[0][1]
  TP = confusion_mat[1][1]
  FN = confusion_mat[1][0]

  TPR = TP/(TP+FN)
  # Specificity or true negative rate
  TNR = TN/(TN+FP) 
  # Precision or positive predictive value
  PPV = TP/(TP+FP)
  # Negative predictive value
  NPV = TN/(TN+FN)
  # Fall out or false positive rate
  FPR = FP/(FP+TN)
  # False negative rate
  FNR = FN/(TP+FN)
  # False discovery rate
  FDR = FP/(TP+FP)

  ll = log_loss(y_test, y_prob) # Its low but means nothing to me. 
  br = brier_score_loss(y_test, y_prob) # Its low but means nothing to me. 
  acc = accuracy_score(y_test, y_predict)
  print(acc)
  auc = roc_auc_score(y_test, y_prob)
  print(auc)
  prc = average_precision_score(y_test, y_prob) 

  data = np.array([np.arange(1)]*1).T

  df_exec = pd.DataFrame(data)

  df_exec["Average Log Likelihood"] = ll
  df_exec["Brier Score Loss"] = br
  df_exec["Accuracy Score"] = acc
  df_exec["ROC AUC Sore"] = auc
  df_exec["Average Precision Score"] = prc
  df_exec["Precision - Bankrupt Firms"] = PPV
  df_exec["False Positive Rate (p-value)"] = FPR
  df_exec["Precision - Healthy Firms"] = NPV
  df_exec["False Negative Rate (recall error)"] = FNR
  df_exec["False Discovery Rate "] = FDR
  df_exec["All Observations"] = TN + TP + FN + FP
  df_exec["Bankruptcy Sample"] = TP + FN
  df_exec["Healthy Sample"] = TN + FP
  df_exec["Recalled Bankruptcy"] = TP + FP
  df_exec["Correct (True Positives)"] = TP
  df_exec["Incorrect (False Positives)"] = FP
  df_exec["Recalled Healthy"] = TN + FN
  df_exec["Correct (True Negatives)"] = TN
  df_exec["Incorrect (False Negatives)"] = FN

  df_exec = df_exec.T[1:]
  df_exec.columns = ["Metrics"]
  return df_exec


