import numpy as np
import pandas as pd
import ast
import time
import json
import dask
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client
import ctypes
import dask.multiprocessing

def trim_memory() -> int:
    """
    helps to fix any memory leaks.
    """
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)
         
def PA1(reviews_csv_path,products_csv_path):
    start = time.time()
    
    #Viggy Code - to be removed after testing
    #client = Client()
    #client.run(trim_memory)
    #client.restart()
    
    #Original code
    client = Client('127.0.0.1:8786')
    client.run(trim_memory)
    client = client.restart()
    print(client)

    # --------------- WRITE YOUR CODE HERE --------------------- 
    print("started my code section")
    
    #Create dataframes from csv files
    reviews_ddf = dd.read_csv(reviews_csv_path)
    products_ddf = dd.read_csv(products_csv_path,dtype={'asin':object})
    
    # Question 1,2 - Missing values
    m1 = reviews_ddf.isnull().mean()*100
    m2 = products_ddf.isnull().mean()*100
    
    #Question 3 - Pearson correlation
    df1 = dd.merge(reviews_ddf[['overall','asin']], products_ddf[['price','asin']], on='asin')
    p3 = df1[['price', 'overall']].corr(method ='pearson')
    
    #Question 4 - Statistics
    s4 = products_ddf.price.describe()
    
    #Question 5 - Supercategories
    df2 = products_ddf
    
    #Custom function to grab first item in list of categories
    def item1(s):
        if isinstance(s, str):
            try:
                first_item = str(ast.literal_eval(s)[0][0])
                return first_item
            except:
                pass
        return '0'

    # Apply custom function to each partition
    df2 = df2.map_partitions(lambda df: df.assign(item1=df['categories'].apply(item1)))

    # Group by the 'item1' supercategory column and count occurrences of each value
    s5 = df2.groupby("item1")['item1'].count(split_out=10)
    
    # Question 6,7 - materialize asins for quick lookup
    asins = products_ddf['asin']
    
    # COMPUTE questions 1-5, asins only for 6-7
    out1, out2, out3, out4, out5, out6 = dd.compute(m1,m2,p3,s4,s5,asins)
    ans1 = out1.round(2).to_dict()
    ans2 = out2.round(2).to_dict()
    ans3 = out3.round(2).at['price', 'overall']
    ans3 = float(ans3)
    out4 = out4.round(2)
    ans4 = {'mean':out4['mean'],'std':out4['std'],'median':out4['50%'],'min':out4['min'],'max':out4['max']}
    ans5 = out5.sort_values(ascending=False).drop(labels=['','0']).to_dict()
    
    #Question 6 - Dangling references (different tables)
    
    #Computed asins placed into set for quick lookup. Then run loop to match reviews_asins vs. products_asins. 
    asins = set(out6)
    for x in reviews_ddf['asin']:
        if x not in asins:
            ans6 = 1
            break
            
    #Question 7 - Dangling references (same table)
    
    #Same as in question 6 except this time we need to cycle through the (key,value) pairs in dictionaries
    found = False  # initialize flag variable to False
    for value in products_ddf['related']:   #loop to iterate through rows of dataframe column
        if not pd.isna(value):
            result = ast.literal_eval(value)
            if isinstance(result, dict):
                for k, v in result.items():      #loop to iterate through dictionary
                    for x in v:                          #loop to iterate through value list
                        if x not in asins:
                            ans7 = 1
                            found = True  # set flag variable to True when dangling reference is found
                            break  # exit inner loop
                    if found:
                        break  # exit outer loop 
                if found:
                    break  # exit outer loop 
        if found:
            break  # exit outer loop 
        
    # ---------------------------------------------------------- 
            
    # assign to each variable the answer to your question. 
    # answers must follow the datatype mentioned
    ans1 = ans1        
    ans2 = ans2
    ans3 = ans3
    ans4 = ans4
    ans5 = ans5
    ans6 = ans6
    ans7 = ans7
 
    # DO NOT MODIFY
    assert type(ans1) == dict, f"answer to question 1 must be a dictionary like {{'reviewerID':0.2, ..}}, got type = {type(ans1)}"
    assert type(ans2) == dict, f"answer to question 2 must be a dictionary like {{'asin':0.2, ..}}, got type = {type(ans2)}"
    assert type(ans3) == float, f"answer to question 3 must be a float like 0.8, got type = {type(ans3)}"
    assert type(ans4) == dict, f"answer to question 4 must be a dictionary like {{'mean':0.4,'max':0.6,'median':0.6...}}, got type = {type(ans4)}"
    assert type(ans5) == dict, f"answer to question 5 must be a dictionary, got type = {type(ans5)}"         
    assert ans6 == 0 or ans6==1, f"answer to question 6 must be 0 or 1, got value = {ans6}" 
    assert ans7 == 0 or ans7==1, f"answer to question 7 must be 0 or 1, got value = {ans7}" 
    
    end = time.time()
    runtime = end-start
    print(f"runtime  = {runtime}s")
    ans_dict = {
        "q1": ans1,
        "q2": ans2,
        "q3": ans3,
        "q4": ans4,
        "q5": ans5,
        "q6": ans6,
        "q7": ans7,
        "runtime": runtime
    }
    with open('results_PA1.json', 'w') as outfile: json.dump(ans_dict, outfile)       
    return runtime  
    
#reviews_csv_path = './user_reviews_Release.csv'
#products_csv_path = './products_Release.csv'
#if __name__ == '__main__':
#    PA1(reviews_csv_path,products_csv_path)