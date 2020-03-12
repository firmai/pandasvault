
# **PandasVault** ⁠— Advanced Pandas Functions and Code Snippets

The only Pandas utility package you would ever need. It has no exotic external dependencies. All functions have been compared and tested with alternatives, only the fastest equivalent functions have been developed and included in this package. The package has more than 20 wrapped functions and 100 snippets. 

[`Github PandasVault Link`](https://github.com/firmai/pandasvault), [`LinkedIn`](https://www.linkedin.com/company/firmai)

You have the option to view this [Readme](https://github.com/firmai/pandasvault/blob/master/pandasvault.md) or run a [Colab](https://colab.research.google.com/drive/1TRKHPGfQnE2yw6_VPBJZ3nZ8lIPQYiuP) Notebook. 

```python
pip install pandasvault
```

If you can identify performance improvements, or improvements in code length and styling, please open a pull request. This package is new, all help and criticisms are appreciated. I would love to hear about any additional function ideas. If you have a **function to contribute** please open an issues tab or email me at d.snow(at)nyu.edu. 

### List of Functions

```python
import pandas as pd
import numpy as np
import pandasvault as pv

"""TABLE PROCESSING"""
df = pv.list_shuff(["target","c","d"],df)
df = pv.reduce_mem_usage(df)

"""TABLE EXPLORATION"""
df = pv.corr_list(df)
df = pv.missing_data(df)

"""FEATURE PROCESSING"""
df = pv.drop_corr(df, thresh=0.1,keep_cols=["target"])
df = pv.replace_small_cat(df,["cat"])
qconstant_col = pv.constant_feature_detect(data=df,threshold=0.9)
df_train, scl = pv.scaler(df,target="target",cols_ignore=["a"],type="MinMax")
df_test = pv.scaler(df_test,scaler=scl,train=False, target="target",cols_ignore=["a"])
df = pv.impute_null_with_tail(df,cols=df.columns)
index,para = pv.outlier_detect(df,"a",threshold=0.5,method="IQR")
df = pv.windsorization(data=df,col='a',para=para,strategy='both')
df = pv.impute_outlier(data=df,col='a', outlier_index=index,strategy='mean')

"""FEATURE EXTRACTION"""
df = pv.auto_dummy(df, unique=3)
df = pv.binarise_empty(df, frac=0.6)
df = pv.polynomials(df, ["a","b"]) 
df = pv.transformations(df,["a","b"])
df = pv.pca_feature(df,variance_or_components=0.80,drop_cols=["target","a"])
df = pv.multiple_lags(df, start=1, end=2,columns=["a","target"])
df = pv.multiple_rolling(df, columns=["a"])
df = pv.date_features(df, date="date_fake")
df['distance_central'] = df.apply(pv.haversine_distance,axis=1)

"""MODEL VALIDATION"""
scores = pv.classification_scores(y_test, y_predict, y_prob)

```


## List of Code


#### [Table Processing](#table-processing)
  -	[<font color="black">Configure Pandas</font>](#configure-pandas)
  -	[<font color="black">Data Frame Formatting</font>](#data-frame-formatting)
  -	[<font color="black">Data Frames for Testing</font>](#data-frames-for-testing)
  -	[<font color="black">Lower Case Columns</font>](#lower-case-columns)
  -	[<font color="black">Front and Back Column Selection</font>](#front-and-back-columns)
  -	[<font color="black">Fast Data Frame Split</font>](#fast-data-frame-split)
  -	[<font color="black">Create Features and Labels List</font>](#create-features-and-labels-list)
  -	[<font color="black">Short Basic Commands</font>](#short-basic-commands)
  -	[<font color="black">Read Commands</font>](#read-commands)
  -	[<font color="black">Create Ordered Categories</font>](#create-ordered-categories)
  -	[<font color="black">Select Columns Based on Regex</font>](#select-columns-based-on-regex)
  -	[<font color="black">Accessing Group of Groupby Object</font>](#accessing-group-of-groupby-object)
  -	[<font color="black">Multiple External Selection Criteria</font>](#multiple-external-selection-criteria)
  -	[<font color="black">Memory Reduction Script</font>](#memory-reduction-script)
  -	[<font color="black">Verify Primary Key</font>](#verify-primary-key)
  -	[<font color="black">Shift Columns to Front</font>](#shift-columns-to-front)
  -	[<font color="black">Multiple Column Assignment</font>](#multiple-column-assignment)
  -	[<font color="black">Method Changing Technique</font>](#method-chaning-event)
  -	[<font color="black">Load Multiple Files</font>](#load-multiple-files)
  -	[<font color="black">Drop Rows and Column Substring</font>](#drop-rows-and-column-substring)
  -	[<font color="black">Explode a Column</font>](#explode-a-column)
  -	[<font color="black">Nest List Back into Column</font>](#nest-list-back-into-column)
  -	[<font color="black">Split Cells with List</font>](#split-cells-with-list)


#### [Table Exploration](#table-exploration)

  -	[<font color="black">Groupby Functionality</font>](#groupby-functionality)
  -	[<font color="black">Cross Correlation Series Without Duplicates](#cross-correlation-series-without-duplicates)
  -	[<font color="black">Missing Data Report</font>](#missing-data-report)
  -	[<font color="black">Duplicated Rows Report</font>](#duplicated-rows-report)
  -	[<font color="black">Skewness</font>](#skewness)


#### [Feature Processing](#feature-processing)

  -	[<font color="black">Remove Correlated Pairs</font>](#remove-correlated-pairs)
  -	[<font color="black">Replace Infrequently Occurring Categories</font>](#replace-infrequently-occuring-categories)
  -	[<font color="black">Quasi-Constant Feature Detection</font>](#quasi-constant-feature-detection)
  -	[<font color="black">Filling Missing Values Separately</font>](#filling-missing-values-separately)
  -	[<font color="black">Conditioned Column Value Replacement</font>](#conditioned-column-value-replacement)
  -	[<font color="black">Remove Non-numeric Values in Data Frame</font>](#remove-non-numeric-values-in-data-frame)
  -	[<font color="black">Feature Scaling, Normalisation, Standardisation</font>](#feature-scaling-normalisation-standardisation)
  -	[<font color="black">Impute Null with Tail Distribution</font>](#impute-null-with-tail-distribution)
  -	[<font color="black">Detect Outliers</font>](#detect-outliers)
  -	[<font color="black">Windzorise Outliers</font>](#windsorize-outliers)
  -	[<font color="black">Drop Outliers</font>](#drop-outliers)
  -	[<font color="black">Impute Outliers</font>](#impute-outliers)

#### [Feature Engineering](#feature-engineering)

  -	[<font color="black">Automated Dummy Encoding</font>](#automate-dummy-encodings)
  -	[<font color="black">Binarise Empty Columns</font>](#binarise-empty-columns)
  -	[<font color="black">Polynomials</font>](#polynomials)
  -	[<font color="black">Transformations</font>](#transformations)
  -	[<font color="black">Genetic Programming</font>](#genetic-programming)
  -	[<font color="black">Principal Component</font>](#principal-component)
  -	[<font color="black">Multiple Lags</font>](#multiple-lags)
  -	[<font color="black">Multiple Rolling</font>](#multiple-rolling)
  -	[<font color="black">Date Features</font>](#data-features)
  -	[<font color="black">Haversine Distance</font>](#havervsine-distance)
  -	[<font color="black">Parse Address</font>](#parse-address)
  -	[<font color="black">Processing Strings in Pandas</font>](#processing-strings-in-pandas)
  -	[<font color="black">Filtering Strings in Pandas</font>](#filtering-strings-in-pandas)


#### [Model Validation](#model-validation)

  -	[<font color="black">Classification Metrics</font>](#classification-metrics)





*If you are running the code <mark>for the first time load this</mark> test dataframe:*


```python
!pip install pandasvault
```


```python
import pandas as pd
import numpy as np
import pandasvault as pv

np.random.seed(1)
"""quick way to create a data frame for testing""" 
df_test = pd.DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd']) \
    .assign(target=lambda x: (x['b']+x['a']/x['d'])*x['c'])
```

<a name="table-processing"></a>

### **Table Processing**


---



---


<a name="configure-pandas"></a>

**>>> Configure Pandas (func)**



---






```python
import pandas as pd

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
    pv.pd_config()

```

<br/>

<a name="data-frame-formatting"></a>

**>>> Data Frame Formatting**

---





```python
df = df_test.copy()
df["number"] = [3,10,1]
```


```python

df_out = (
  df.style.format({"a":"${:.2f}", "target":"${:.5f}"})
 .hide_index()
 .highlight_min("a", color ="red")
 .highlight_max("a", color ="green")
 .background_gradient(subset = "target", cmap ="Blues")
 .bar("number", color = "lightblue", align = "zero")
 .set_caption("DF with different stylings")
) ; df_out

```

```See Colab for Output```


<br/>
<a name="data-frame-formatting"></a>

**>>> Data Frames For Testing**

---




```python
df1 = pd.util.testing.makeDataFrame() # contains random values
print("Contains missing values")
df2 = pd.util.testing.makeMissingDataframe() # contains missing values
print("Contains datetime values")
df3 = pd.util.testing.makeTimeDataFrame() # contains datetime values
print("Contains mixed values")
df4 = pd.util.testing.makeMixedDataFrame(); df4.head() # contains mixed values

```

    Contains missing values
    Contains datetime values
    Contains mixed values





 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>foo1</td>
      <td>2009-01-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>foo2</td>
      <td>2009-01-02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>foo3</td>
      <td>2009-01-05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>foo4</td>
      <td>2009-01-06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>foo5</td>
      <td>2009-01-07</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="lower-case-columns"></a>


**>>> Lower Case Columns**


---






```python
## Lower-case all DataFrame column names 
df = df_test.copy() ; df
df.columns = ["A","BGs","c","dag","Target"]
```


```python
df.columns = map(str.lower, df.columns); df
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>bgs</th>
      <th>c</th>
      <th>dag</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="front-and-back-columns"></a>

**>>> Front and Back Column Selection**



---






```python
df = df_test.copy(); df
```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 




```python
def front(self, n):
    return self.iloc[:, :n]

def back(self, n):
    return self.iloc[:, -n:]

pd.back = back
pd.front = front

pd.back(df,2)
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>d</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="fast-data-frame-split"></a>

**>>> Fast Data Frame Split**



---







```python
df = df_test.copy(); df
```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 




```python
test =  df.sample(frac=0.4)
train = df[~df.isin(test)].dropna(); train
```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="create-features-and-labels-list"></a>

**>>> Create Features and Labels List**



---






```python
df = df_test.head()
y = 'target'
X = [name for name in df.columns if name not in [y, 'd']]
print('y =', y)
print('X =', X)
```

    y = target
    X = ['a', 'b', 'c']

<br/>

<a name="short-basic-commands"></a>

**>>> Short Basic Commands**



---





```python
df = df_test.copy()
df["category"] = np.where( df["target"]>1, "1",  "0")
df["k"] = df["category"].astype(str) +": " + df["d"].round(1).astype(str) 
df = df.append(df, ignore_index=True) ; df.head()
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>category</th>
      <th>k</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>1</td>
      <td>1: -1.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>0</td>
      <td>0: -0.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>0</td>
      <td>0: -2.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>1</td>
      <td>1: -1.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>0</td>
      <td>0: -0.8</td>
    </tr>
  </tbody>
</table>
 




```python
"""set display width, col_width etc for interactive pandas session""" 
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 20)
pd.set_option('display.max_rows', 100)
           
"""when you have an excel sheet with spaces in column names"""
df.columns = [c.lower().replace(' ', '_') for c in df.columns]

"""Add prefix to all columns"""
df.add_prefix("1_")

"""Add suffix to all columns"""
df.add_suffix("_Z")

"""Droping column where missing values are above a threshold"""
df.dropna(thresh = len(df)*0.95, axis = "columns") 

"""Given a dataframe df to filter by a series ["a","b"]:""" 
df[df['category'].isin(["1","0"])]

"""filter by multiple conditions in a dataframe df"""
df[(df['a'] >1) & (df['b'] <1)]

"""filter by conditions and the condition on row labels(index)"""
df[(df.a > 0) & (df.index.isin([0, 1]))]

"""regexp filters on strings (vectorized), use .* instead of *"""
df[df.category.str.contains(r'.*[0-9].*')]

"""logical NOT is like this"""
df[~df.category.str.contains(r'.*[0-9].*')]

"""creating complex filters using functions on rows"""
df[df.apply(lambda x: x['b'] > x['c'], axis=1)]

"""Pandas replace operation"""
df["a"].round(2).replace(0.87, 17, inplace=True)
df["a"][df["a"] < 4] = 19

"""Conditionals and selectors"""
df.loc[df["a"] > 1, ["a","b","target"]]

"""Selecting multiple column slices"""
df.iloc[:, np.r_[0:2, 4:5]] 

"""apply and map examples"""
df[["a","b","c"]].applymap(lambda x: x+1)

"""add 2 to row 3 and return the series"""
df[["a","b","c"]].apply(lambda x: x[0]+2,axis=0)

"""add 3 to col A and return the series"""
df.apply(lambda x: x['a']+1,axis=1)

""" Split delimited values in a DataFrame column into two new columns """
df['new1'], df['new2'] = zip(*df['k'].apply(lambda x: x.split(': ', 1)))

""" Doing calculations with DataFrame columns that have missing values
  In example below, swap in 0 for df['col1'] cells that contain null """ 
df['new3'] = np.where(pd.isnull(df['b']),0,df['a']) + df['c']

""" Exclude certain data type or include certain data types """
df.select_dtypes(exclude=['O','float'])
df.select_dtypes(include=['int'])

"""one liner to normalize a data frame""" 
(df[["a","b"]] - df[["a","b"]].mean()) / (df[["a","b"]].max() - df[["a","b"]].min())

"""groupby used like a histogram to obtain counts on sub-ranges of a variable, pretty handy""" 
df.groupby(pd.cut(df.a, range(0, 1, 2))).size()

"""use a local variable use inside a query of pandas using @"""
mean = df["a"].mean()
df.query("a > @mean")

"""Calculate the % of missing values in each column"""
df.isna().mean() 

"""Calculate the % of missing values in each row"""
rows = df.isna().mean(axis=1) ; df.head()

```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>category</th>
      <th>k</th>
      <th>new1</th>
      <th>new2</th>
      <th>new3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19.0</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>1</td>
      <td>1: -1.1</td>
      <td>1</td>
      <td>-1.1</td>
      <td>18.4718</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19.0</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>0</td>
      <td>0: -0.8</td>
      <td>0</td>
      <td>-0.8</td>
      <td>20.7448</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.0</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>0</td>
      <td>0: -2.1</td>
      <td>0</td>
      <td>-2.1</td>
      <td>20.4621</td>
    </tr>
    <tr>
      <th>3</th>
      <td>19.0</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>1</td>
      <td>1: -1.1</td>
      <td>1</td>
      <td>-1.1</td>
      <td>18.4718</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19.0</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>0</td>
      <td>0: -0.8</td>
      <td>0</td>
      <td>-0.8</td>
      <td>20.7448</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="read-commands"></a>

**>>>  Read Commands**






---




```python
df = pd.util.testing.makeMixedDataFrame()
df.to_csv("data.csv") ; df
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>foo1</td>
      <td>2009-01-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>foo2</td>
      <td>2009-01-02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>foo3</td>
      <td>2009-01-05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>foo4</td>
      <td>2009-01-06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>foo5</td>
      <td>2009-01-07</td>
    </tr>
  </tbody>
</table>
 




```python
"""To avoid Unnamed: 0 when loading a previously saved csv with index"""
"""To parse dates"""
"""To set data types"""

df_out = pd.read_csv("data.csv", index_col=0,
                 parse_dates=['D'],
                 dtype={"c":"category", "B":"int64"}).set_index("D")

"""Copy data to clipboard; like an excel copy and paste
df = pd.read_clipboard()
"""

"""Read table from website
df = pd.read_html(url, match="table_name")
"""

""" Read pdf into dataframe ()
!pip install tabula
from tabula import read_pdf
df = read_pdf('test.pdf', pages='all')
"""
df_out.head()
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
    <tr>
      <th>D</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2009-01-01</th>
      <td>0.0</td>
      <td>0</td>
      <td>foo1</td>
    </tr>
    <tr>
      <th>2009-01-02</th>
      <td>1.0</td>
      <td>1</td>
      <td>foo2</td>
    </tr>
    <tr>
      <th>2009-01-05</th>
      <td>2.0</td>
      <td>0</td>
      <td>foo3</td>
    </tr>
    <tr>
      <th>2009-01-06</th>
      <td>3.0</td>
      <td>1</td>
      <td>foo4</td>
    </tr>
    <tr>
      <th>2009-01-07</th>
      <td>4.0</td>
      <td>0</td>
      <td>foo5</td>
    </tr>
  </tbody>
</table>
 

<br/>


<a name="create-ordered-categories"></a>

**>>> Create Ordered Categories**


---



```python
df = df_test.copy()
df["cats"] = ["bad","good","excellent"]; df
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>cats</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>bad</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>good</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>excellent</td>
    </tr>
  </tbody>
</table>
 




```python
import pandas as pd
from pandas.api.types import CategoricalDtype

print("Let's create our own categorical order.")
cat_type = CategoricalDtype(["bad", "good", "excellent"], ordered = True)
df["cats"] = df["cats"].astype(cat_type)

print("Now we can use logical sorting.")
df = df.sort_values("cats", ascending = True)

print("We can also filter this as if they are numbers.")
df[df["cats"] > "bad"]

```

    Let's create our own categorical order.
    Now we can use logical sorting.
    We can also filter this as if they are numbers.





 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>cats</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>good</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>excellent</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="select-columns-based-on-regex"></a>

**>>> Select Columns Based on Regex**


---




```python
df = df_test.head(); df
df.columns = ["a_l", "b_l", "c_r","d_r","target"]  ; df
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a_l</th>
      <th>b_l</th>
      <th>c_r</th>
      <th>d_r</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 




```python
df_out = df.filter(regex="_l",axis=1) ; df_out
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a_l</th>
      <th>b_l</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="accessing-group-of-groupby-object"></a>

**>>> Accessing Group of Groupby Object**


---




```python
df = df_test.copy()
df = df.append(df, ignore_index=True)
df["groupie"] = ["falcon","hawk","hawk","eagle","falcon","hawk"]; df
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>groupie</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>falcon</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>hawk</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>hawk</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>eagle</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>falcon</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>hawk</td>
    </tr>
  </tbody>
</table>
 




```python
gbdf = df.groupby("groupie")
hawk = gbdf.get_group("hawk").mean(); hawk
```




    a         0.5012
    b        -0.9334
    c         1.5563
    d        -1.6272
    target   -2.3938
    dtype: float64


<br/>

<a name="multiple-external-selection-criteria"></a>

**>>> Multiple External Selection Criteria**





---




```python
df = df_test.copy()
```


```python
cr1 = df["a"] > 0
cr2 = df["b"] < 0
cr3 = df["c"] > 0
cr4 = df["d"] >-1

df[cr1 & cr2 & cr3 & cr4]

```




 
<table  class="dataframe">
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
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="memory-reduction-script"></a>

**>>> Memory Reduction Script (func)**



---






```python
df = df_test.copy(); df
```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 




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
df_out = pv.reduce_mem_usage(df); df_out
```

    Memory usage of dataframe is 0.00 MB
    Memory usage after optimization is: 0.00 MB
    Decreased by 36.3%





 
<table  class="dataframe">
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
      <td>1.6240</td>
      <td>-0.6118</td>
      <td>-0.5283</td>
      <td>-1.0732</td>
      <td>1.1230</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8652</td>
      <td>-2.3008</td>
      <td>1.7451</td>
      <td>-0.7612</td>
      <td>-6.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3191</td>
      <td>-0.2494</td>
      <td>1.4619</td>
      <td>-2.0605</td>
      <td>-0.5908</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="verify-primary-key"></a>

**>>> Verify Primary Key (func)**




---




```python
df = df_test.copy()
df["first_d"] = [0,1,2]
df["second_d"] = [4,1,9] ; df
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>first_d</th>
      <th>second_d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>2</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
 




```python
def verify_primary_key(df, column_list):
    '''Verify if columns in column list can be treat as primary key'''

    return df.shape[0] == df.groupby(column_list).size().reset_index().shape[0]

verify_primary_key(df, ["first_d","second_d"])
```




    True


<br/>

<a name="shift-columns-to-front"></a>

**>>> Shift Columns to Front (func)**


---




```python
df = df_test.copy(); df
```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 




```python
def list_shuff(items, df):
    "Bring a list of columns to the front"
    cols = list(df)
    for i in range(len(items)):
        cols.insert(i, cols.pop(cols.index(items[i])))
    df = df.loc[:, cols]
    df.reset_index(drop=True, inplace=True)
    return df

df_out = pv.list_shuff(["target","c","d"],df); df_out
```




 
<table  class="dataframe">
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
      <td>1.1227</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.6243</td>
      <td>-0.6118</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-5.9994</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>0.8654</td>
      <td>-2.3015</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.5910</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>0.3190</td>
      <td>-0.2494</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="multiple-column-assignment"></a>

**>>> Multiple Column Assignments**


---




```python
df = df_test.copy(); df
```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 




```python
df_out = (df.assign(stringed = df["a"].astype(str),
            ounces = df["b"]*12,#                                     this will allow yo set a title
            galons = lambda df: df["a"]/128)
           .query("b > -1")
           .style.set_caption("Average consumption")) ; df_out

```




<table id="T_6d6167c4_627f_11ea_a8db_0242ac1c0002" ><caption>Average consumption</caption><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >a</th>        <th class="col_heading level0 col1" >b</th>        <th class="col_heading level0 col2" >c</th>        <th class="col_heading level0 col3" >d</th>        <th class="col_heading level0 col4" >target</th>        <th class="col_heading level0 col5" >stringed</th>        <th class="col_heading level0 col6" >ounces</th>        <th class="col_heading level0 col7" >galons</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_6d6167c4_627f_11ea_a8db_0242ac1c0002level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_6d6167c4_627f_11ea_a8db_0242ac1c0002row0_col0" class="data row0 col0" >1.624</td>
                        <td id="T_6d6167c4_627f_11ea_a8db_0242ac1c0002row0_col1" class="data row0 col1" >-0.6118</td>
                        <td id="T_6d6167c4_627f_11ea_a8db_0242ac1c0002row0_col2" class="data row0 col2" >-0.5282</td>
                        <td id="T_6d6167c4_627f_11ea_a8db_0242ac1c0002row0_col3" class="data row0 col3" >-1.073</td>
                        <td id="T_6d6167c4_627f_11ea_a8db_0242ac1c0002row0_col4" class="data row0 col4" >1.123</td>
                        <td id="T_6d6167c4_627f_11ea_a8db_0242ac1c0002row0_col5" class="data row0 col5" >1.6243453636632417</td>
                        <td id="T_6d6167c4_627f_11ea_a8db_0242ac1c0002row0_col6" class="data row0 col6" >-7.341</td>
                        <td id="T_6d6167c4_627f_11ea_a8db_0242ac1c0002row0_col7" class="data row0 col7" >0.01269</td>
            </tr>
            <tr>
                        <th id="T_6d6167c4_627f_11ea_a8db_0242ac1c0002level0_row1" class="row_heading level0 row1" >2</th>
                        <td id="T_6d6167c4_627f_11ea_a8db_0242ac1c0002row1_col0" class="data row1 col0" >0.319</td>
                        <td id="T_6d6167c4_627f_11ea_a8db_0242ac1c0002row1_col1" class="data row1 col1" >-0.2494</td>
                        <td id="T_6d6167c4_627f_11ea_a8db_0242ac1c0002row1_col2" class="data row1 col2" >1.462</td>
                        <td id="T_6d6167c4_627f_11ea_a8db_0242ac1c0002row1_col3" class="data row1 col3" >-2.06</td>
                        <td id="T_6d6167c4_627f_11ea_a8db_0242ac1c0002row1_col4" class="data row1 col4" >-0.591</td>
                        <td id="T_6d6167c4_627f_11ea_a8db_0242ac1c0002row1_col5" class="data row1 col5" >0.31903909605709857</td>
                        <td id="T_6d6167c4_627f_11ea_a8db_0242ac1c0002row1_col6" class="data row1 col6" >-2.992</td>
                        <td id="T_6d6167c4_627f_11ea_a8db_0242ac1c0002row1_col7" class="data row1 col7" >0.002492</td>
            </tr>
    </tbody></table>

<br/>


<a name="method-chaning-event"></a>

**>>> Method Chaining Technique**



---





```python
df = df_test.copy()
df[df>df.mean()]  = None ; df
```




 
<table  class="dataframe">
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
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.5282</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-2.0601</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
 




```python
# with line continuation character
df_out = df.dropna(subset=["b","c"],how="all") \
.loc[df["a"]>0] \
.round(2) \
.groupby(["target","b"]).max() \
.unstack() \
.fillna(0) \
.rolling(1).sum() \
.reset_index() \
.stack() \
.ffill().bfill() 

df_out
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>a</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
    </tr>
    <tr>
      <th></th>
      <th>b</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">0</th>
      <th>-2.3</th>
      <td>0.87</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-6.0</td>
    </tr>
    <tr>
      <th></th>
      <td>0.87</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-6.0</td>
    </tr>
  </tbody>
</table>
 

<br/>


<a name="load-multiple-files"></a>

**>>> Load Multiple Files**



---





```python
import os
os.makedirs("folder",exist_ok=True,); df_test.to_csv("folder/first.csv",index=False) ; df_test.to_csv("folder/last.csv",index=False)
```


```python
import glob
files = glob.glob('folder/*.csv')
dfs = [pd.read_csv(fp) for fp in files]
df_out = pd.concat(dfs)
```


```python
df_out
```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 

<br/>


<a name="drop-rows-and-column-substring"></a>

**>>> Drop Rows with Column Substring**


---




```python
df = df_test.copy()
df["string_feature"] = ["1xZoo", "Safe7x", "bat4"]; df
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>string_feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>1xZoo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>Safe7x</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>bat4</td>
    </tr>
  </tbody>
</table>
 




```python
substring = ["xZ","7z", "tab4"]

df_out = df[~df.string_feature.str.contains('|'.join(substring))]; df_out
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>string_feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>Safe7x</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>bat4</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="explode-a-column"></a>

**>>> Unnest (Explode) a Column**



---




```python
df = df_test.head()
df["g"] = [[str(a)+lista for a in range(4)] for lista in ["a","b","c"]]; df
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>g</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>[0a, 1a, 2a, 3a]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>[0b, 1b, 2b, 3b]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>[0c, 1c, 2c, 3c]</td>
    </tr>
  </tbody>
</table>
 




```python
df_out = df.explode("g"); df_out.iloc[:5,:]
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>g</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>0a</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>1a</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>2a</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>3a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>0b</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="nest-list-back-into-column"></a>

**>>> Nest List Back into Column**


---




```python
### Run above example first 
df = df_out.copy()
```


```python
df_out['g'] = df_out.groupby(df_out.index)['g'].agg(list); df_out.head()
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>g</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>[0a, 1a, 2a, 3a]</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>[0a, 1a, 2a, 3a]</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>[0a, 1a, 2a, 3a]</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>[0a, 1a, 2a, 3a]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>[0b, 1b, 2b, 3b]</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="split-cells-with-list"></a>

**>>> Split Cells With Lists**


---




```python
df = df_test.head()
df["g"] = [",".join([str(a)+lista for a in range(4)]) for lista in ["a","b","c"]]; df
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>g</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>0a,1a,2a,3a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>0b,1b,2b,3b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>0c,1c,2c,3c</td>
    </tr>
  </tbody>
</table>
 




```python
df_out = df.assign(g = df["g"].str.split(",")).explode("g"); df_out.head()
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>g</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>0a</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>1a</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>2a</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>3a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>0b</td>
    </tr>
  </tbody>
</table>
 

<br/>


<a name="table-exploration"></a>

### **Table Exploration**



---



---



<a name="groupby-functionality"></a>

**>>> Groupby Functionality**

---




```python
df = df_test.head() 
df["gr"] = [1, 1 , 0] ;df
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>gr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
 




```python
In [34]: gb.<TAB>  # noqa: E225, E999
gb.agg        gb.boxplot    gb.cummin     gb.describe   gb.filter     
gb.get_group  gb.height     gb.last       gb.median     gb.ngroups    
gb.plot       gb.rank       gb.std        gb.transform  gb.aggregate  
gb.count      gb.cumprod    gb.dtype      gb.first      gb.nth
gb.groups     gb.hist       gb.max        gb.min        gb.gender        
gb.prod       gb.resample   gb.sum        gb.var        gb.ohlc  
gb.apply      gb.cummax     gb.cumsum     gb.fillna          
gb.head       gb.indices    gb.mean       gb.name            
gb.quantile   gb.size       gb.tail       gb.weight

```


```python
df_out = df.groupby('gr').agg([np.sum, np.mean, np.std]); df_out.iloc[:,:8]
```



<table  class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">a</th>
      <th colspan="3" halign="left">b</th>
      <th colspan="2" halign="left">c</th>
    </tr>
    <tr>
      <th></th>
      <th>sum</th>
      <th>mean</th>
      <th>std</th>
      <th>sum</th>
      <th>mean</th>
      <th>std</th>
      <th>sum</th>
      <th>mean</th>
    </tr>
    <tr>
      <th>gr</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.3190</td>
      <td>0.3190</td>
      <td>NaN</td>
      <td>-0.2494</td>
      <td>-0.2494</td>
      <td>NaN</td>
      <td>1.4621</td>
      <td>1.4621</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.4898</td>
      <td>1.2449</td>
      <td>0.5367</td>
      <td>-2.9133</td>
      <td>-1.4566</td>
      <td>1.1949</td>
      <td>1.2166</td>
      <td>0.6083</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="cross-correlation-series-without-duplicates"></a>

**>>> Cross Correlation Series Without Duplicates (func)**





---




```python
df = df_test.copy(); df
```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 




```python
def corr_list(df):

  return  (df.corr()
          .unstack()
          .sort_values(kind="quicksort",ascending=False)
          .drop_duplicates().iloc[1:]); df_out
          
pv.corr_list(df)
```




    b       target    0.9215
    a       d         0.6605
            target    0.3206
    b       a        -0.0724
    c       d        -0.1764
            b        -0.4545
    target  d        -0.4994
    c       target   -0.7647
    b       d        -0.7967
    a       c        -0.8555
    dtype: float64


<br/>

<a name="missing-data-report"></a>

**>>> Missing Data Report (func)**



---





```python
df = df_test.copy()
df[df>df.mean()]  = None ; df
```




 
<table  class="dataframe">
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
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.5282</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-2.0601</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
 




```python

def missing_data(data):
    "Create a dataframe with a percentage and count of missing values"
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

pv.df_out = missing_data(df); df_out
```




<table  class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">a</th>
      <th colspan="3" halign="left">b</th>
      <th colspan="3" halign="left">c</th>
      <th colspan="3" halign="left">d</th>
      <th colspan="3" halign="left">target</th>
    </tr>
    <tr>
      <th></th>
      <th>sum</th>
      <th>mean</th>
      <th>std</th>
      <th>sum</th>
      <th>mean</th>
      <th>std</th>
      <th>sum</th>
      <th>mean</th>
      <th>std</th>
      <th>sum</th>
      <th>mean</th>
      <th>std</th>
      <th>sum</th>
      <th>mean</th>
      <th>std</th>
    </tr>
    <tr>
      <th>gr</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.3190</td>
      <td>0.3190</td>
      <td>NaN</td>
      <td>-0.2494</td>
      <td>-0.2494</td>
      <td>NaN</td>
      <td>1.4621</td>
      <td>1.4621</td>
      <td>NaN</td>
      <td>-2.0601</td>
      <td>-2.0601</td>
      <td>NaN</td>
      <td>-0.5910</td>
      <td>-0.5910</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.4898</td>
      <td>1.2449</td>
      <td>0.5367</td>
      <td>-2.9133</td>
      <td>-1.4566</td>
      <td>1.1949</td>
      <td>1.2166</td>
      <td>0.6083</td>
      <td>1.6072</td>
      <td>-1.8342</td>
      <td>-0.9171</td>
      <td>0.2204</td>
      <td>-4.8767</td>
      <td>-2.4384</td>
      <td>5.0361</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="duplicated-rows-report"></a>

**>>> Duplicated Rows Report**



---




```python
df = df_test.copy()
df["a"].iloc[2] = df["a"].iloc[1]
df["b"].iloc[2] = df["b"].iloc[1] ; df
```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 




```python
# Get a report of all duplicate records in a dataframe, based on specific columns
df_out = df[df.duplicated(['a', 'b'], keep=False)] ; df_out

```




 
<table  class="dataframe">
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
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="skewness"></a>

**>>> Skewness (func)**


---




```python
df = df_test.copy(); df
```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 




```python
from scipy.stats import skew

def display_skewness(data):
    '''show skewness information

        Parameters
        ----------
        data: pandas dataframe

        Return
        ------
        df: pandas dataframe
    '''
    numeric_cols = data.columns[data.dtypes != 'object'].tolist()
    skew_value = []

    for i in numeric_cols:
        skew_value += [skew(data[i])]
    df = pd.concat(
        [pd.Series(numeric_cols), pd.Series(data.dtypes[data.dtypes != 'object'].apply(lambda x: str(x)).values)
            , pd.Series(skew_value)], axis=1)
    df.columns = ['var_name', 'col_type', 'skew_value']

    return df

display_skewness(df)

```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>var_name</th>
      <th>col_type</th>
      <th>skew_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>float64</td>
      <td>0.1963</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>float64</td>
      <td>-0.6210</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>float64</td>
      <td>-0.6659</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>float64</td>
      <td>-0.5427</td>
    </tr>
    <tr>
      <th>4</th>
      <td>target</td>
      <td>float64</td>
      <td>-0.5418</td>
    </tr>
  </tbody>
</table>
 




<a name="feature-processing"></a>

### **Feature Processing**


---



---


<a name="remove-correlated-pairs"></a>

**>>> Remove Correlated Pairs (func)**



---







```python
df= df_test.copy(); df
```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 




```python
def drop_corr(df, thresh=0.99,keep_cols=[]):
    df_corr = df.corr().abs()
    upper = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(np.bool))
    to_remove = [column for column in upper.columns if any(upper[column] > thresh)] ## Change to 99% for selection
    to_remove = [x for x in to_remove if x not in keep_cols]
    df_corr = df_corr.drop(columns = to_remove)
    return df.drop(to_remove,axis=1)

df_out = pv.drop_corr(df, thresh=0.1,keep_cols=["target"]); df_out
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="replace-infrequently-occuring-categories"></a>

**>>> Replace Infrequently Occuring Categories**



---




```python
df = df_test.copy()
df = df.append([df]*2)
df["cat"] = ["bat","bat","rat","mat","mat","mat","mat","mat","mat"]; df
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>bat</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>bat</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>rat</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>mat</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>mat</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>mat</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>mat</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>mat</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>mat</td>
    </tr>
  </tbody>
</table>
 




```python

def replace_small_cat(df, columns, thresh=0.2, term="other"):
  for col in columns:

    # Step 1: count the frequencies
    frequencies = df[col].value_counts(normalize = True)

  # Step 2: establish your threshold and filter the smaller categories

    small_categories = frequencies[frequencies < thresh].index

    df[col] = df[col].replace(small_categories, "Other")
    
  return df

df_out = pv.replace_small_cat(df,["cat"]); df_out.head()

```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>bat</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>bat</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>Other</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>mat</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>mat</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="quasi-constant-feature-detection"></a>

**>>> Quasi-Constant Features Detection (func)**


---




```python
df = df_test.copy()
df["a"] = 3 
```


```python

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
qconstant_col = pv.constant_feature_detect(data=df,threshold=0.9)
df_out = df.drop(qconstant_col, axis=1) ; df_out
```

    1  variables are found to be almost constant





 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 




```python
### I will take care of outliers separately
```
<br/>

<a name="filling-missing-values-separately"></a>

**>>> Filling Missing Values Separately**


---




```python
df = df_test.copy()
df[df>df.mean()]  = None ; df
```




 
<table  class="dataframe">
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
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.5282</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-2.0601</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
 




```python
# Clean up missing values in multiple DataFrame columns
dict_fill = {'a': 4,
              'b': 3,
              'c': 5,
              'd': 9999,
              'target': "False"}
df = df.fillna(dict_fill) ;df

```




 
<table  class="dataframe">
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
      <td>4.0000</td>
      <td>3.0000</td>
      <td>-0.5282</td>
      <td>9999.0000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>5.0000</td>
      <td>9999.0000</td>
      <td>-5.999</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>3.0000</td>
      <td>5.0000</td>
      <td>-2.0601</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="conditioned-column-value-replacement"></a>

**>>> Conditioned Column Value Replacement**


---




```python
df = df_test.copy(); df
```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 




```python
# Set DataFrame column values based on other column values
df.loc[(df['a'] >1 ) & (df['c'] <0), ['target']] = np.nan ;df
```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 

<br/>


<a name="remove-non-numeric-values-in-data-frame"></a>

**>>> Remove Non-numeric Values in Data Frame**




---




```python
df = df_test.copy().assign(target=lambda row: row["a"].round(4).astype(str)+"SC"+row["b"].round(4).astype(str))
df["a"] = "TI4560L" + df["a"].round(4).astype(str) ; df
```




 
<table  class="dataframe">
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
      <td>TI4560L1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.6243SC-0.6118</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TI4560L0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>0.8654SC-2.3015</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TI4560L0.319</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>0.319SC-0.2494</td>
    </tr>
  </tbody>
</table>
 




```python
df_out = df.replace('[^0-9]+', '', regex=True); df_out
```




 
<table  class="dataframe">
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
      <td>456016243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1624306118</td>
    </tr>
    <tr>
      <th>1</th>
      <td>456008654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>0865423015</td>
    </tr>
    <tr>
      <th>2</th>
      <td>45600319</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>031902494</td>
    </tr>
  </tbody>
</table>
 

<br/>


<a name="feature-scaling-normalisation-standardisation"></a>

**>>> Feature Scaling, Normalisation, Standardisation (func)**



---




```python
df= df_test.copy(); df
```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 




```python
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

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

df_out_train, scl = pv.scaler(df,target="target",cols_ignore=["a"],type="MinMax")
df_out_test = pv.scaler(df_test,scaler=scl,train=False, target="target",cols_ignore=["a"]); df_out_test


```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>0.8234</td>
      <td>0.0000</td>
      <td>0.76</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>0.0000</td>
      <td>1.0000</td>
      <td>1.00</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>1.0000</td>
      <td>0.8756</td>
      <td>0.00</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="impute-null-with-tail-distribution"></a>

**>>> Impute Null with Tail Distribution (func)**



---




```python
df = df_test.copy()
df[df>df.mean()]  = None ; df
```




 
<table  class="dataframe">
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
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.5282</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-2.0601</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
 




```python
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
  

df_out = pv.impute_null_with_tail(df,cols=df.columns); df_out
```




 
<table  class="dataframe">
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
      <td>1.7512</td>
      <td>NaN</td>
      <td>-0.5282</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-2.0601</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="detect-outliers"></a>

**>>> Detect Outliers (func)**






---




```python
df = df_test.copy(); df
```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 




```python
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
    

index,para = pv.outlier_detect(df,"a",threshold=0.5,method="IQR")
print('Upper bound:',para[0],'\nLower bound:',para[1])
```

    Num of outlier detected: 1
    Proportion of outlier detected 0.3333333333333333
    Upper bound: 1.5712030633954956 
    Lower bound: 0.2658967957893529

<br/>

<a name="windsorize-outliers"></a>

**>>> Windsorize Outliers (func)**






---




```python
# RUN above example first
df = df_test.copy(); df

```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 




```python
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


df_out = pv.windsorization(data=df,col='a',para=para,strategy='both'); df_out
```




 
<table  class="dataframe">
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
      <td>1.5712</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="drop-outiers"></a>

**>>> Drop Outliers**






---




```python
## run the top two examples
df = df_test.copy(); df
```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 




```python
df_out = df[~index] ; df_out
```




 
<table  class="dataframe">
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
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="impute-outiers (func)"></a>

**>>> Impute Outliers**






---




```python
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
  
df_out = pv.impute_outlier(data=df,col='a', outlier_index=index,strategy='mean'); df_out
```




 
<table  class="dataframe">
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
      <td>0.9363</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 



<a name="feature-engineering"></a>

### **Feature Engineering**



---



---



<a name="automate-dummy-encodings"></a>

**>>> Automated Dummy (one-hot) Encoding (func)**



---




```python
df = df_test.copy()
df["e"] = np.where(df["c"]> df["a"], 1,  2)
```


```python
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

df_out = pv.auto_dummy(df, unique=3); df_out
```

 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>e_1</th>
      <th>e_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="binarise-empty-columns"></a>

**>>> Binarise Empty Columns (func)**


---




```python
df = df_test.copy()
df[df>df.mean()]  = None ; df
```




 
<table  class="dataframe">
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
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.5282</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-2.0601</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
 




```python
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

df_out = pv.binarise_empty(df, frac=0.6); df_out
```

    b
    c
    d
    target





 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b_0</th>
      <th>b_1</th>
      <th>c_0</th>
      <th>c_1</th>
      <th>d_0</th>
      <th>d_1</th>
      <th>target_0</th>
      <th>target_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="polynomials"></a>

**>>> Polynomials (func)**



---




```python
df = df_test.copy(); df
```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 




```python
def polynomials(df, feature_list):
  for feat in feature_list:
    for feat_two in feature_list:
      if feat==feat_two:
        continue
      else:
       df[feat+"/"+feat_two] = df[feat]/(df[feat_two]-df[feat_two].min()) #zero division guard
       df[feat+"X"+feat_two] = df[feat]*(df[feat_two])

  return df

df_out = pv.polynomials(df, ["a","b"]) ; df_out
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>a/b</th>
      <th>aXb</th>
      <th>b/a</th>
      <th>bXa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>0.9613</td>
      <td>-0.9937</td>
      <td>-0.4687</td>
      <td>-0.9937</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>inf</td>
      <td>-1.9918</td>
      <td>-4.2124</td>
      <td>-1.9918</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>0.1555</td>
      <td>-0.0796</td>
      <td>-inf</td>
      <td>-0.0796</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="transformations"></a>

**>>> Transformations (func)**



---




```python
df = df_test.copy(); df
```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 




```python
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

df_out = pv.transformations(df,["a","b"]); df_out.iloc[:,:8]
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>a_POWER_2</th>
      <th>b_POWER_2</th>
      <th>a_LOG_p_one_abs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>1.7038</td>
      <td>2.8554</td>
      <td>0.8352</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>0.2985</td>
      <td>0.0000</td>
      <td>0.4359</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>0.0000</td>
      <td>4.2114</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="genetic-programming"></a>

**>>> Genetic Programming**



---




```python
! pip install gplearn
```

    Collecting gplearn
    [?25l  Downloading https://files.pythonhosted.org/packages/43/6b/ee38cd74b32ad5056603aabbef622f9691f19d0869574dfc610034f18662/gplearn-0.4.1-py3-none-any.whl (41kB)
    [K     |████████████████████████████████| 51kB 2.5MB/s 
    [?25hRequirement already satisfied: scikit-learn>=0.20.0 in /usr/local/lib/python3.6/dist-packages (from gplearn) (0.22.1)
    Requirement already satisfied: joblib>=0.13.0 in /usr/local/lib/python3.6/dist-packages (from gplearn) (0.14.1)
    Requirement already satisfied: numpy>=1.11.0 in /usr/local/lib/python3.6/dist-packages (from scikit-learn>=0.20.0->gplearn) (1.17.5)
    Requirement already satisfied: scipy>=0.17.0 in /usr/local/lib/python3.6/dist-packages (from scikit-learn>=0.20.0->gplearn) (1.4.1)
    Installing collected packages: gplearn
    Successfully installed gplearn-0.4.1



```python
df = df_test.copy(); df
```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 




```python
from gplearn.genetic import SymbolicTransformer
function_set = ['add', 'sub', 'mul', 'div',
                'sqrt', 'log', 'abs', 'neg', 'inv','tan']

gp = SymbolicTransformer(generations=800, population_size=200,
                         hall_of_fame=100, n_components=10,
                         function_set=function_set,
                         parsimony_coefficient=0.0005,
                         max_samples=0.9, verbose=1,
                         random_state=0, n_jobs=6)

gen_feats = gp.fit_transform(df.drop("target", axis=1), df["target"]); df.iloc[:,:8]
df_out = pd.concat((df,pd.DataFrame(gen_feats, columns=["gen_"+str(a) for a in range(gen_feats.shape[1])])),axis=1); df_out.iloc[:,:8]

```

        |   Population Average    |             Best Individual              |
    ---- ------------------------- ------------------------------------------ ----------
     Gen   Length          Fitness   Length          Fitness      OOB Fitness  Time Left
       0    10.14             0.91       22                1                0     43.36m





 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>gen_0</th>
      <th>gen_1</th>
      <th>gen_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>-1.8292</td>
      <td>-2.6469</td>
      <td>0.5059</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>-3.5190</td>
      <td>99.1619</td>
      <td>3.6243</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>-1.4668</td>
      <td>1.3677</td>
      <td>3.1826</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="principal-component"></a>

**>>> Prinicipal Component Features (func)**




---




```python
df =df_test.copy(); df
```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 




```python
from sklearn.decomposition import PCA, IncrementalPCA

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

df_out = pv.pca_feature(df,variance_or_components=0.80,drop_cols=["target","a"]); df_out

```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>a</th>
      <th>PCA_1</th>
      <th>PCA_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.1227</td>
      <td>1.6243</td>
      <td>-1.2944</td>
      <td>-0.7684</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-5.9994</td>
      <td>0.8654</td>
      <td>1.5375</td>
      <td>-0.4537</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.5910</td>
      <td>0.3190</td>
      <td>-0.2431</td>
      <td>1.2220</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="multiple-lags"></a>

**>>> Multiple Lags (func)**




---




```python
df = df_test.copy(); df
```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 




```python
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

df_out = pv.multiple_lags(df, start=1, end=2,columns=["a","target"]); df_out
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>a_t_1</th>
      <th>target_t_1</th>
      <th>a_t_2</th>
      <th>target_t_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>1.6243</td>
      <td>1.1227</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>0.8654</td>
      <td>-5.9994</td>
      <td>1.6243</td>
      <td>1.1227</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="multiple-rolling"></a>

**>>> Multiple Rolling (func)**


---




```python
df = df_test.copy(); df
```




 
<table  class="dataframe">
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
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
    </tr>
  </tbody>
</table>
 




```python
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

df_out = pv.multiple_rolling(df, columns=["a"]); df_out
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>a_2_mean</th>
      <th>a_2_std</th>
      <th>a_3_mean</th>
      <th>a_3_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>1.2449</td>
      <td>0.5367</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>0.5922</td>
      <td>0.3863</td>
      <td>0.9363</td>
      <td>0.6555</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="data-features"></a>

**>>> Date Features**



---




```python
df = df_test.copy()
df["date_fake"] = pd.date_range(start="2019-01-03", end="2019-01-06", periods=len(df)); df
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>date_fake</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>2019-01-03 00:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>2019-01-04 12:00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>2019-01-06 00:00:00</td>
    </tr>
  </tbody>
</table>
 




```python
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

df_out = date_features(df, date="date_fake"); df_out.iloc[:,:8]
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>date_fake</th>
      <th>date_fake_month</th>
      <th>date_fake_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>2019-01-03 00:00:00</td>
      <td>1</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>2019-01-04 12:00:00</td>
      <td>1</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>2019-01-06 00:00:00</td>
      <td>1</td>
      <td>2019</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="havervsine-distance"></a>

**>>> Haversine Distance (Location Feature) (func)**


---




```python
df = df_test.copy()
df["latitude"] = [39, 35 , 20]
df["longitude"]=  [-77, -40 , -10 ]
```


```python
from math import sin, cos, sqrt, atan2, radians
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

df['distance_central'] = df.apply(pv.haversine_distance,axis=1); df.iloc[:,4:]
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>distance_central</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.1227</td>
      <td>39</td>
      <td>-77</td>
      <td>6702.7127</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-5.9994</td>
      <td>35</td>
      <td>-40</td>
      <td>4583.5988</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.5910</td>
      <td>20</td>
      <td>-10</td>
      <td>4141.6783</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="parse-address"></a>

**>>> Parse Address**



---




```python
df = df_test.copy()
df["addr"] = pd.Series([
            'Washington, D.C. 20003',
            'Brooklyn, NY 11211-1755',
            'Omaha, NE 68154' ]) ; df
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>target</th>
      <th>addr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.6243</td>
      <td>-0.6118</td>
      <td>-0.5282</td>
      <td>-1.0730</td>
      <td>1.1227</td>
      <td>Washington, D.C....</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.8654</td>
      <td>-2.3015</td>
      <td>1.7448</td>
      <td>-0.7612</td>
      <td>-5.9994</td>
      <td>Brooklyn, NY 112...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.3190</td>
      <td>-0.2494</td>
      <td>1.4621</td>
      <td>-2.0601</td>
      <td>-0.5910</td>
      <td>Omaha, NE 68154</td>
    </tr>
  </tbody>
</table>
 




```python
regex = (r'(?P<city>[A-Za-z ]+), (?P<state>[A-Z]{2}) (?P<zip>\d{5}(?:-\d{4})?)')  

df.addr.str.replace('.', '').str.extract(regex)
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city</th>
      <th>state</th>
      <th>zip</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Washington</td>
      <td>DC</td>
      <td>20003</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brooklyn</td>
      <td>NY</td>
      <td>11211-1755</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Omaha</td>
      <td>NE</td>
      <td>68154</td>
    </tr>
  </tbody>
</table>
 


<br/>

<a name="processing-strings-in-pandas"></a>

**>>> Processing Strings in Pandas**



---




```python
df = pd.util.testing.makeMixedDataFrame()
df["C"] = df["C"] + " " + df["C"] ; df
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>foo1 foo1</td>
      <td>2009-01-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>foo2 foo2</td>
      <td>2009-01-02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>foo3 foo3</td>
      <td>2009-01-05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>foo4 foo4</td>
      <td>2009-01-06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>foo5 foo5</td>
      <td>2009-01-07</td>
    </tr>
  </tbody>
</table>
 




```python
"""convert column to UPPERCASE"""

col_name = "C"
df[col_name].str.upper()

"""count string occurence in each row"""
df[col_name].str.count(r'\d') # counts number of digits

"""count # o chars in each row"""
df[col_name].str.count('o') # counts number of digits

"""split rows"""
s = pd.Series(["this is a regular sentence", "https://docs.p.org", np.nan])
s.str.split()

"""this creates new columns with the different split values (instead of lists)"""
s.str.split(expand=True)  

"""limit the number of splits to 1, and start spliting from the rights side"""
s.str.rsplit("/", n=1, expand=True) 

```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>this is a regula...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https:/</td>
      <td>docs.p.org</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
 

<br/>


<a name="filtering-strings-in-pandas"></a>

**>>> Filtering Strings in Pandas**




---




```python
df = pd.util.testing.makeMixedDataFrame()
df["C"] = df["C"] + " " + df["C"] ; df
```




 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>foo1 foo1</td>
      <td>2009-01-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>foo2 foo2</td>
      <td>2009-01-02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>foo3 foo3</td>
      <td>2009-01-05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>foo4 foo4</td>
      <td>2009-01-06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>foo5 foo5</td>
      <td>2009-01-07</td>
    </tr>
  </tbody>
</table>
 




```python
col_name = "C"

"""check if a certain word/pattern occurs in each row"""
df[col_name].str.contains('oo')  # returns True/False for each row

"""find occurences"""
df[col_name].str.findall(r'[ABC]\d') # returns a list of the found occurences of the specified pattern for each row

"""replace Weekdays by abbrevations (e.g. Monday --> Mon)"""
df[col_name].str.replace(r'(\w+day\b)', lambda x: x.groups[0][:3]) # () in r'' creates a group with one element, which we acces with x.groups[0]

"""create dataframe from regex groups (str.extract() uses first match of the pattern only)"""
df[col_name].str.extract(r'(\d?\d):(\d\d)')
df[col_name].str.extract(r'(?P<hours>\d?\d):(?P<minutes>\d\d)')
df[col_name].str.extract(r'(?P<time>(?P<hours>\d?\d):(?P<minutes>\d\d))')

"""if you want to take into account ALL matches in a row (not only first one):"""
df[col_name].str.extractall(r'(\d?\d):(\d\d)') # this generates a multiindex with level 1 = 'match', indicating the order of the match

df[col_name].replace('\n', '', regex=True, inplace=True)

"""remove all the characters after &# (including &#) for column - col_1"""
df[col_name].replace(' &#.*', '', regex=True, inplace=True)

"""remove white space at the beginning of string"""
df[col_name] = df[col_name].str.lstrip()

```

<a name="model-validation"></a>

## **Model Validation**


---



---



<a name="classification-metrics"></a>
**>>> Classification Metrics (func)**



---




```python
y_test = [0, 1, 1, 1, 0]
y_predict = [0, 0, 1, 1, 1]
y_prob = [0.2,0.6,0.7,0.7,0.9]
```


```python
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

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


met = pv.classification_scores(y_test, y_predict, y_prob); met
```

    0.6
    0.5





 
<table  class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Metrics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Average Log Likelihood</th>
      <td>0.7500</td>
    </tr>
    <tr>
      <th>Brier Score Loss</th>
      <td>0.2380</td>
    </tr>
    <tr>
      <th>Accuracy Score</th>
      <td>0.6000</td>
    </tr>
    <tr>
      <th>ROC AUC Sore</th>
      <td>0.5000</td>
    </tr>
    <tr>
      <th>Average Precision Score</th>
      <td>0.6944</td>
    </tr>
    <tr>
      <th>Precision - Bankrupt Firms</th>
      <td>0.6667</td>
    </tr>
    <tr>
      <th>False Positive Rate (p-value)</th>
      <td>0.5000</td>
    </tr>
    <tr>
      <th>Precision - Healthy Firms</th>
      <td>0.5000</td>
    </tr>
    <tr>
      <th>False Negative Rate (recall error)</th>
      <td>0.3333</td>
    </tr>
    <tr>
      <th>False Discovery Rate</th>
      <td>0.3333</td>
    </tr>
    <tr>
      <th>All Observations</th>
      <td>5.0000</td>
    </tr>
    <tr>
      <th>Bankruptcy Sample</th>
      <td>3.0000</td>
    </tr>
    <tr>
      <th>Healthy Sample</th>
      <td>2.0000</td>
    </tr>
    <tr>
      <th>Recalled Bankruptcy</th>
      <td>3.0000</td>
    </tr>
    <tr>
      <th>Correct (True Positives)</th>
      <td>2.0000</td>
    </tr>
    <tr>
      <th>Incorrect (False Positives)</th>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>Recalled Healthy</th>
      <td>2.0000</td>
    </tr>
    <tr>
      <th>Correct (True Negatives)</th>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>Incorrect (False Negatives)</th>
      <td>1.0000</td>
    </tr>
  </tbody>
</table>
 


