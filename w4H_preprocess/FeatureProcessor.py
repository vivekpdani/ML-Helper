import pickle as pk
import pandas as pd

def convertCat2OneHot(dataframe, cat_col_name, possible_values=[]):
    '''
    dataframe: Pandas Dataframe having dataset
    cat_col_name: Name of the column in dataframe which has categorical feature.
    possible_values: list of unique possible values that <cat_col_name> column can have. 
                     Why do we need this ?
                     Ans: Its not necessary that dataframe has all the possible unique values that the <cat_col_name> column can have.
    '''
    for d in possible_values:
        #Add dummy rows accounting for cases when some of the possible values for the column are not present in dataframe
        dataframe = dataframe.append(pd.Series(), ignore_index=True)
        dataframe.loc[len(dataframe)-1,cat_col_name]=d
    colmn = pd.Series(dataframe[cat_col_name])
    one_hot_colmn=pd.get_dummies(colmn, prefix=cat_col_name)
    dataframe = dataframe.join(one_hot_colmn)
    dataframe = dataframe.drop(columns=[cat_col_name])
    df_size = len(dataframe)
    dataframe = dataframe.drop([i for i in range(df_size-1,df_size-len(possible_values)-1,-1)])
    return dataframe
    


def convertCat2IntByCorrelation(dataframe, cat_col_name, label_name, base_path="", positive_label=1):
    '''
    Convert a categorical feature to 2 float features.
    1) Percentage of positives "pct_postv"
    2) Confidence for that instance "conf_postv"
    
    e.g.) Let categorical feature be <Month> and the prediction label be 0 or 1 stating whether it rained in that month or not.
    Lets take month of "May", so conf_postv = `Number of records in dataset that have <Month>="May"`
    positive_records = `Number of records that says it rained in month of "May"`
    pct_postv = positive_records/conf_postv (this gives probability of raining given that its month of May)
    
    From dataframe, we remove this column of <Month> and add 2 new columns, <Month_pct_postv> which is what is the probability of raining for that month
    and 2nd is <Month_conf_postv> which says how many records did our dataset have to back our probability calculation.
    
    dataframe: Pandas Dataframe having dataset and label both
    cat_col_name: Name of the column in dataframe which has categorical feature.
    label_name, positive_label=1
    '''
    
    qty_postv = dataframe[cat_col_name][dataframe[label_name] == positive_label].value_counts().sort_values(ascending=False)
    qty_total = dataframe[cat_col_name].value_counts()

    pct_postv={}
    for col_value, qty_postv_val in qty_postv.iteritems():
        pct_postv[col_value]= qty_postv_val/qty_total[col_value]
        
    dataframe[cat_col_name+"_pct_postv"]=0
    dataframe[cat_col_name+"_conf_postv"]=0

    for k, v in pct_postv.items():
        dataframe.loc[dataframe[cat_col_name] == k, cat_col_name+'_pct_postv'] = pct_postv[k]
        dataframe.loc[dataframe[cat_col_name] == k, cat_col_name+'_conf_postv'] = qty_total[k]
    
    dataframe[cat_col_name+"_pct_postv"].fillna(0)
    dataframe = dataframe.drop(columns=[cat_col_name])
    
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    with open(base_path+cat_col_name+' pct_postv.pk', 'wb') as handle:
        pk.dump(pct_postv, handle)  
        
    with open(base_path+cat_col_name+' qty_total.pk', 'wb') as handle:
        pk.dump(qty_total, handle)
    
    return dataframe


def convertCat2IntByCorrelTestData(dataframe, cat_col_name):
    '''
    Convert a categorical feature to 2 float features.
    1) Percentage of positives "pct_postv"
    2) Confidence for that instance "conf_postv"
    
    e.g.) Let categorical feature be <Month> and the prediction label be 0 or 1 stating whether it rained in that month or not.
    Lets take month of "May", so conf_postv = `Number of records in dataset that have <Month>="May"`
    positive_records = `Number of records that says it rained in month of "May"`
    pct_postv = positive_records/conf_postv (this gives probability of raining given that its month of May)
    
    From dataframe, we remove this column of <Month> and add 2 new columns, <Month_pct_postv> which is what is the probability of raining for that month
    and 2nd is <Month_conf_postv> which says how many records did our dataset have to back our probability calculation.
    
    dataframe: Pandas Dataframe having dataset and label both
    cat_col_name: Name of the column in dataframe which has categorical feature.
    '''
    global base_path
    
    pct_postv=''
    qty_total=''
    
    with open(base_path+cat_col_name+' pct_postv.pk', 'rb') as handle:
        pct_postv = pk.load(handle)  
        
    with open(base_path+cat_col_name+' qty_total.pk', 'rb') as handle:
        qty_total = pk.load(handle)
    
    dataframe[cat_col_name+"_pct_postv"]=0
    dataframe[cat_col_name+"_conf_postv"]=0

    for k, v in pct_postv.items():
        dataframe.loc[dataframe[cat_col_name] == k, cat_col_name+'_pct_postv'] = pct_postv[k]
        dataframe.loc[dataframe[cat_col_name] == k, cat_col_name+'_conf_postv'] = qty_total[k]
    
    dataframe[cat_col_name+"_pct_postv"].fillna(0)
    dataframe = dataframe.drop(columns=[cat_col_name])
    
    return dataframe
