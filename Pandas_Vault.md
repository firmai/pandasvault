
# Pandas Vault

**>>>Create Test Dataframe**


```
import pandas as pd
import numpy as np
"""quick way to create a data frame for testing""" 
df = pd.DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd']) \
    .assign(target=lambda x: (x['b']+x['a']/x['d'])*x['c'])

```


```
df
```
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.642987</td>
      <td>-1.592816</td>
      <td>0.469016</td>
      <td>-0.529240</td>
      <td>-0.177238</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.527028</td>
      <td>0.225137</td>
      <td>0.228500</td>
      <td>0.656817</td>
      <td>0.234792</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.444146</td>
      <td>0.946610</td>
      <td>-0.309655</td>
      <td>1.373406</td>
      <td>-0.393262</td>
    </tr>
  </tbody>
</table>

&nbsp;
&nbsp;

### **Data Processing**

>>> Memory Reduction Script


```
# Input
df_in = df.copy(); df_in
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.431641</td>
      <td>0.081116</td>
      <td>0.594727</td>
      <td>-0.905762</td>
      <td>0.331787</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.416992</td>
      <td>1.874023</td>
      <td>-0.530273</td>
      <td>-0.374268</td>
      <td>-3.001953</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.646484</td>
      <td>-2.167969</td>
      <td>-1.365234</td>
      <td>0.474365</td>
      <td>7.695312</td>
    </tr>
  </tbody>
</table>


```
# Code
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
df_out = reduce_mem_usage(df_in)
```

    Memory usage of dataframe is 0.00 MB
    Memory usage after optimization is: 0.00 MB
    Decreased by 0.0%



```
# output
df_out
```



<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.431641</td>
      <td>0.081116</td>
      <td>0.594727</td>
      <td>-0.905762</td>
      <td>0.331787</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.416992</td>
      <td>1.874023</td>
      <td>-0.530273</td>
      <td>-0.374268</td>
      <td>-3.001953</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.646484</td>
      <td>-2.167969</td>
      <td>-1.365234</td>
      <td>0.474365</td>
      <td>7.695312</td>
    </tr>
  </tbody>
</table>



**>>> Missing Data Report**


```
df_in = df.copy()
df_in[df_in>df_in.mean()] = None ; df_in
```


<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.642987</td>
      <td>-1.592816</td>
      <td>NaN</td>
      <td>-0.52924</td>
      <td>-0.177238</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.309655</td>
      <td>NaN</td>
      <td>-0.393262</td>
    </tr>
  </tbody>
</table>


```
def missing_data(data):
    "Create a dataframe with a percentage and count of missing values"
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

df_out = missing_data(df_in); df_out
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>d</th>
      <td>2</td>
      <td>66.666667</td>
    </tr>
    <tr>
      <th>c</th>
      <td>2</td>
      <td>66.666667</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2</td>
      <td>66.666667</td>
    </tr>
    <tr>
      <th>a</th>
      <td>2</td>
      <td>66.666667</td>
    </tr>
    <tr>
      <th>target</th>
      <td>1</td>
      <td>33.333333</td>
    </tr>
  </tbody>
</table>


**>>> Shift Columns to Front**


```
df_in = df.copy(); df_in
```


<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.642987</td>
      <td>-1.592816</td>
      <td>0.469016</td>
      <td>-0.529240</td>
      <td>-0.177238</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.527028</td>
      <td>0.225137</td>
      <td>0.228500</td>
      <td>0.656817</td>
      <td>0.234792</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.444146</td>
      <td>0.946610</td>
      <td>-0.309655</td>
      <td>1.373406</td>
      <td>-0.393262</td>
    </tr>
  </tbody>
</table>


```
def ListShuff(items, df):
    "Bring a list of columns to the front"
    cols = list(df)
    for i in range(len(items)):
        cols.insert(i, cols.pop(cols.index(items[i])))
    df = df.loc[:, cols]
    df.reset_index(drop=True, inplace=True)
    return df

df_out = ListShuff(["target","c","d"],df_in); df_out
```

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>c</th>
      <th>d</th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.177238</td>
      <td>0.469016</td>
      <td>-0.529240</td>
      <td>-0.642987</td>
      <td>-1.592816</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.234792</td>
      <td>0.228500</td>
      <td>0.656817</td>
      <td>0.527028</td>
      <td>0.225137</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.393262</td>
      <td>-0.309655</td>
      <td>1.373406</td>
      <td>0.444146</td>
      <td>0.946610</td>
    </tr>
  </tbody>
</table>


