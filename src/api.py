import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def segmentation(path, model):
    
    #load data
    df=pd.read_csv(path, header=None)
    header=['age', 'class of worker', 'detailed industry recode', 'detailed occupation recode', 'education', 
            'wage per hour', 'enroll in edu inst last wk', 'marital stat', 'major industry code', 
            'major occupation code', 'race', 'hispanic origin', 'sex', 'member of a labor union', 
            'reason for unemployment', 'full or part time employment stat', 'capital gains', 
            'capital losses', 'dividends from stocks', 'tax filer stat', 'region of previous residence', 
            'state of previous residence', 'detailed household and family stat', 
            'detailed household summary in household', 'instance weight', 'migration code-change in msa', 
            'migration code-change in reg', 'migration code-move within reg', 'live in this house 1 year ago', 
            'migration prev res in sunbelt', 'num persons worked for employer', 'family members under 18', 
            'country of birth father', 'country of birth mother', 'country of birth self', 'citizenship', 
            'own business or self employed', "fill inc questionnaire for veteran's admin", 'veterans benefits', 
            'weeks worked in year', 'year', 'label']
    if len(df.columns)!=40:
        df.columns=header
        df.drop(['instance weight','label'],inplace=True,axis=1)
    else:
        header.remove('label')
        header.remove('instance weight')
        df.columns=header
    
    # data cleaning and feature engineering
    df_new=df.copy()
    df_new.drop('year',axis=1,inplace=True)
    df_new.drop(["fill inc questionnaire for veteran's admin",'reason for unemployment',
                 'enroll in edu inst last wk','state of previous residence','region of previous residence',
                 'migration prev res in sunbelt','member of a labor union'],axis=1,inplace=True)

    for count, col in enumerate(df_new.columns):
        df_new[col] = df_new[col].replace('?','Not in universe')
    
    df_new['citizenship_cov']=np.where(df['citizenship']=='Native- Born in the United States',1,0)
    df_new['hispanic_cov']=np.where(df['hispanic origin']=='All other',0,1)
    df_new['race_cov']=np.where(df['race']=='White',1,0)
    df_new.drop(['citizenship','hispanic origin','race'],axis=1,inplace=True)
    
    remap_cat_dict = {
    'Children':0,
    'Less than 1st grade': 0,   
    '1st 2nd 3rd or 4th grade': 1,
    '5th or 6th grade': 1,
    '7th and 8th grade': 1,
    '9th grade': 1,
    '10th grade': 1,
    '11th grade': 1,
    'High school graduate': 2,
    '12th grade no diploma': 2,
    'Some college but no degree': 3,
    'Bachelors degree(BA AB BS)':3,
    'Masters degree(MA MS MEng MEd MSW MBA)':4,
    'Associates degree-occup /vocational':4,
    'Associates degree-academic program':4,
    'Doctorate degree(PhD EdD)':5,
    'Prof school degree (MD DDS DVM LLB JD)':5
    }
    df_new.education = df_new.education.map(remap_cat_dict).astype('category')
    df_new.education.astype('int')
    
    df_new['capital']=df_new['capital gains']-df_new['capital losses']+df_new['dividends from stocks']
    df_new['invest']=np.where(df_new['capital']!=0,1,0)
    df_new.drop(['capital','capital gains','capital losses','dividends from stocks'],axis=1,inplace=True)
    
    df_new['wage']=np.where(df_new['wage per hour']!=0,1,0)
    df_new.drop('wage per hour',axis=1,inplace=True)
    
    df_new.drop(['detailed household and family stat','migration code-move within reg',
                 'detailed industry recode','detailed occupation recode'],axis=1,inplace=True)
    
    df_new.drop(['country of birth father', 'country of birth mother', 'country of birth self'],axis=1,inplace=True)
    df_new.drop(['migration code-change in msa','migration code-change in reg',
                 'live in this house 1 year ago', 'citizenship_cov', 'wage'],axis=1,inplace=True)
    
    multi_cols=['class of worker', 'marital stat', 'major industry code',
       'major occupation code', 'full or part time employment stat', 'tax filer stat', 
       'detailed household summary in household',
       'own business or self employed', 'veterans benefits','family members under 18']
    bin_cols=['label','hispanic_cov', 'race_cov', 'invest', 'sex']
    num_cols=['age','education','num persons worked for employer','weeks worked in year']
    
    df_new = pd.get_dummies(data = df_new,columns = multi_cols)
    le = LabelEncoder()
    for i in bin_cols:
        df_new[i] = le.fit_transform(df_new[i]) 
    std = StandardScaler()
    df_new[num_cols] = std.fit_transform(df_new[num_cols])
    
   
    predictions_rf   = model.predict(df_new)
    probabilities_rf = model.predict_proba(df_new)
    
    return predictions_rf