def columns_datatypes(myDataFrame):
    import numpy as np
    import pandas as pd
    """
    This function goal is to generate a dictionnary which build some metadatas
    about the column datatypes from the entry DataFrame.

    RESULT : a dictionnary with 2 keys:
    - 'ColumnTypes' gives a summary of each column datatype recognized within
    the DataFrame : its values are the lists of the columns whose type match
    with the dictionary key.
    - 'CountTypes' gives the number of columns matching with each datatype.

    PARAMS :
    - 'MyDataFrame' : The entry DataFrame


    EXAMPLE :
    columns_datatypes(df)
    >>
    {'ColumnType': {
                        'float64': ['CustomerID'],
                        'int64': ['Quantity'],
                        'object': ['InvoiceNo','StockCode','Description',
                        'InvoiceDate','UnitPrice','Country']
                    },
     'CountType': {
                        'float64': 1,
                        'int64': 1,
                        'object': 6
                    }
    }
    """
    #We list the datatypes recognized by a pandas DataFrame.
    L_dtypes = ['float64','int64','object','datetime64[ns]','float32','bool','int8']
    L_emptyKeys = []

    dict_dtypes = {'float64' :[],
               'int64' :[],
               'object' :[],
               'datetime64[ns]' :[],
               'float32' :[],
               'bool' :[],
               'int8' :[],
               'dtypeReject' : []}

    present_types = {}

    df_dtypes = pd.DataFrame(myDataFrame.dtypes,columns=['datatype'])

    for columnName in df_dtypes.index :
        datum = df_dtypes.loc[columnName,'datatype']
        datum = str(datum)

        if datum in L_dtypes:
            dict_dtypes[datum].append(columnName)

        else :
            dict_dtypes['dtypeReject'].append(columnName)

    for datatype in dict_dtypes:
        if len(dict_dtypes[datatype])>0:
            present_types[datatype]=len(dict_dtypes[datatype])

        else:
            L_emptyKeys.append(datatype)

    for datatypekey in L_emptyKeys :
        del dict_dtypes[datatypekey]

    return({'ColumnType':dict_dtypes , 'CountType':present_types})

def content_analysis(myDataFrame):
    import numpy as np
    import pandas as pd
    """
    This function goal is to generate a DataFrame which contains metadatas about
    a DataFrame content.
    IMPORTANT : This function uses the output of the 'columns_datatypes'
    function so you need both functions for using this one.

    RESULT : A DataFrame which contains metadatas about the entry DataFrame :
    - 'nullCount' : number of missing values
    - 'percent_missing' : percent of missing values compared to the DataFrame
    lenght
    - 'Unique' : number of unique values.
    - 'Unique_percent' : percent of unique values compared to the DataFrame
    lenght
    - 'Datatype' : datatype recocgnized for each columns
    - 'col_missing>10','col_missing>20','col_missing>30','col_missing>40' : 4 columns
    which contains 'r' if the percent of missing values are respectively under 10%,
    20%, 30% and 40% or 'g' on the other hand : this is useful for plotting
    the missing values by columns with colors.

    PARAMS :
    - 'MyDataFrame' : The entry DataFrame
    """
    Empty_List = []
    Unique_values = []
    myDataFrame_len = len(myDataFrame)

    for column in list(myDataFrame):
        Empty_List.append(len(myDataFrame[column].unique()))

    #We build the Dataframe of unique values percents :
    UniqueDataframe = pd.DataFrame(Empty_List,index=list(myDataFrame),columns=['Unique'])
    UniqueDataframe['Unique_percent'] = (UniqueDataframe['Unique']/myDataFrame_len)*100
    DataTypeDataFrame = pd.DataFrame([],index=list(myDataFrame),columns=['Datatype'])

    for datatype in columns_datatypes(myDataFrame)['ColumnType'].keys():
        columnList = columns_datatypes(myDataFrame)['ColumnType'][datatype]

        for columnName in columnList:
            DataTypeDataFrame.set_value(columnName,'Datatype',datatype)

    #We build the summary DataFrame :
    SummaryDataFrame = pd.DataFrame(myDataFrame.isnull().sum(),columns=['nullCount'])
    SummaryDataFrame['percent_missing']=np.nan
    SummaryDataFrame['percent_missing']=(SummaryDataFrame['nullCount']/myDataFrame_len)*100
    L_null = SummaryDataFrame[SummaryDataFrame['nullCount'] == myDataFrame_len].index.tolist()
    SummaryDataFrame = pd.concat([SummaryDataFrame,UniqueDataframe,DataTypeDataFrame],axis=1)

    for criterium in range(10,41,10):
        missing_col_criterium = "col_missing>%s"%criterium
        SummaryDataFrame[missing_col_criterium] = np.where(SummaryDataFrame['percent_missing']>criterium, 'r', 'g')

    return(SummaryDataFrame)

def identify_my_quantiles(my_rfmd_DataFrame,my_quantile_column):
    """
    This function goal is to build an identifier DataFrame (which will be used
    in the "scatter_plot" function).
    Based on RFMD quartiles, the function returns a "main_category" and a
    "color" columns.

    RESULT : A DataFrame which contains a "main_category" and a "color" columns

    PARAMS :
    - 'my_rfmd_DataFrame' refers to the dataframe we want to identify : it must
    contain the column we want to flag as category and color
    - 'my_quantile_column' refers to the column we want to flag
    """
    import pandas as pd

    L_quantiles =list(my_rfmd_DataFrame[my_quantile_column])
    df_quantiles = pd.DataFrame(L_quantiles,columns=['color'])
    df_quantiles['main_category']=df_quantiles['color'].astype(str)

    return df_quantiles

def RScore(x,param,dictionary):
    """
    This function goal is to build a column of quartiles (1,2,3 or 4) based on a
    continuous feature values.
    The more the feature is high, the more the quartile returned is low

    RESULT : A new DataFrame column which contains the quartiles applied to the
    continuous feature.

    PARAMS :
    - 'x' refers to the feature
    - 'param' refers to the key we want to use in our dictionnary of quartiles
    - 'dictionary' refers to a dictionary of quartiles

    example :
    quantiles = {
                'recency': {0.25: 16.0, 0.5: 60.0, 0.75: 149.0}
                }
    my_dataframe['r_quartile'] = my_dataframe['recency'].apply(RScore, args=('recency',quantiles,))
    """
    if x <= dictionary[param][0.25]:
        return 1
    elif x <= dictionary[param][0.50]:
        return 2
    elif x <= dictionary[param][0.75]:
        return 3
    else:
        return 4

def FMScore(x,param,dictionary):
    """
    This function goal is to build a column of quartiles (1,2,3 or 4) based on a
    continuous feature values.
    The more the feature is high, the more the quartile returned is high

    RESULT : A new DataFrame column which contains the quartiles applied to the
    continuous feature.

    PARAMS :
    - 'x' refers to the feature
    - 'param' refers to the key we want to use in our dictionnary of quartiles
    - 'dictionary' refers to a dictionary of quartiles

    example :
    quantiles = {
    'density': {0.25: 73.0, 0.5: 133.2, 0.75: 230.53125},
    'monetary_value': {0.25: 265.9249999999999, 0.5: 580.05, 0.75: 1404.515},
    'recency': {0.25: 16.0, 0.5: 60.0, 0.75: 149.0}
                }
    my_dataframe['d_quartile'] = my_dataframe['density'].apply(RScore, args=('density',quantiles,))
    """
    if x <= dictionary[param][0.25]:
        return 4
    elif x <= dictionary[param][0.50]:
        return 3
    elif x <= dictionary[param][0.75]:
        return 2
    else:
        return 1

def apply_quartiles(myDataFrame,myMappingColumnDict,myQuartileFunction,QuantilesDict,bool_convert_in_percent):
    """
    This function goal is to compute quartiles from a DataFrame continuous
    columns.
    IMPORTANT : You need a function which compute quartiles (RScore or FMScore)
    if you want to use this fuction.

    RESULT : The entry DataFrame with quartiles computed

    PARAMS :
    - 'myDataFrame' refers to the entry DataFrame
    - 'myMappingColumnDict' refers to a dictionary which maps the entry DataFrame
    columns names with the new names we want in the function output
    - 'myQuartileFunction' refers to the function we want to use in order to
    compute the quariles (RScore or FMScore)
    - 'QuantilesDict' refers to the quantiles dictionnary we want to use in order
    to apply the transform.
    - 'bool_convert_in_percent' is a boolean attributes that specify if we want or
    not convert the quartiles in percents.
    """
    myTempDataFrame = myDataFrame.copy()
    for column in myMappingColumnDict.keys():
        new_column_name = myMappingColumnDict[column]
        myDataFrame[new_column_name] = myDataFrame[column].apply(myQuartileFunction, args=(column,QuantilesDict,))

        if bool_convert_in_percent == True:
            myDataFrame[new_column_name] = 100*(1/ myDataFrame[new_column_name])
            myDataFrame[new_column_name][myTempDataFrame[column]==0] = 0

    del myTempDataFrame

    return myDataFrame

def values_to_col(myDataFrame,myColumnList,bool_with_old_col_name):
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd
    """
    This function goal is to treat categorical features in a pandas DataFrame
    list of columns:
    From a categorical column 'CC' which contains 'N' attributes
    [att1, att2, att3,..., attn ] we create N new vectors/features/columns :
        - row to row, if the category was present at the intersection of 'CC'
        and the row,then the value at the intersection of the row and the new
        column is 1
        - else, the value at the intersection of the row and the new column is 0
    The relation between rows and columns is kept


    RESULT : The entry DataFrame with the new categorical vectors :
    2 new columns are also created :
        - 'created_columns' : a column with a list of all the new created columns
        - 'dict_mapping' : a column with a dictionary which maps the old columns
        with the columns they generated

    PARAMS :
    - 'myDataFrame' refers to the DataFrame we interest in
    - 'myColumnList' refers to the list of columns (the list can have only one
    value but it must be a list) we want to vectorize
    - 'bool_with_old_col_name' is a boolean attribute that specify if we want to
    keep the old columns names or not :
        --> example : with old names, new columns are :
        CC_att1, CC_att2, CC_att3,..., CC_attn
        --> example : without old names : att1, att2, att3,..., attn
    """
    created_columns = []
    dict_mapping = {}
    for column in myColumnList:
        #Missing values filling
        myDataFrame[column].fillna('none', inplace=True)
        newFeatures = []
        corpus = myDataFrame[column]
        vectorizer = CountVectorizer(min_df=1,max_df=1.0)

        #Construction of the row/words Matrix
        X = vectorizer.fit_transform(corpus).toarray()
        feature_names = vectorizer.get_feature_names()

        for feature in feature_names:
            if bool_with_old_col_name==True:
                newFeatureName = '%s_%s'%(column,feature)
            else:
                newFeatureName = feature

            newFeatures.append(newFeatureName)
            created_columns.append(newFeatureName)

            if column in dict_mapping :
                dict_mapping[column].append(newFeatureName)
            else:
                dict_mapping[column] = [newFeatureName]

        #Construction of the row/words DataFrame
        myfeaturedf = pd.DataFrame(X,columns=newFeatures)
        myDataFrame = pd.concat([myDataFrame, myfeaturedf], axis=1, join_axes=[myfeaturedf.index])
        myDataFrame['created_columns']=[created_columns]*len(myDataFrame)
        myDataFrame['dict_mapping']=[dict_mapping]*len(myDataFrame)

    return myDataFrame

def vectors_metadata(myVectorizedDataFrame,argument):
    """
    This function goal is to build some metadatas that summarizes the effects
    of the 'values_to_col' myQuartileFunction
    IMPORTANT : it is not possible to use it before the use of the
    'values_to_col' function because we need the 'created_columns' and
    'dict_mapping' columns

    RESULT : A summary of the vectors creation resulting from the 'values_to_col'
    function

    PARAMS :
    - 'myVectorizedDataFrame' refers to the output DataFrame of the
    - 'values_to_col' function'argument' refers to the granularity of the
    metatadas built from the entry DataFrame:
        --> if 'argument'='summary' : the metadatas will concern the impact of the
        'values_to_col' function for each old column in the list 'myColumnList'

        --> if 'argument'='global' : the metadatas will concern the impact of the
        'values_to_col' function for each old column in the list 'myColumnList'
        and each new column created
    """
    L_vectorized_features = myVectorizedDataFrame['created_columns'][0]
    vectorsMapping = myVectorizedDataFrame['dict_mapping'][0]

    df_mapping_vect = pd.DataFrame(columns=['key','value'])
    for key in vectorsMapping.keys():

        df_temp = pd.DataFrame(vectorsMapping[key],columns=['value'])
        df_temp['key'] = key
        frames = [df_mapping_vect,df_temp]
        result = pd.concat(frames)
        df_mapping_vect = result

    df_VectorAnalysis = VectorAnalysis(myVectorizedDataFrame[L_vectorized_features])
    df_VectorAnalysis['value'] = df_VectorAnalysis.index
    df_VectorAnalysis = pd.merge(df_VectorAnalysis, df_mapping_vect, how='left', on=['value'])

    L_metadata_cols = ['number_of_vectors','max_1_Count','min_1_Count','average_1_Count','%10_quantile_1_Count',
                '%25_quantile_1_Count','%30_quantile_1_Count','median_quantile_1_Count']

    for col in L_metadata_cols :
        df_VectorAnalysis[col] = np.nan

    for key in vectorsMapping.keys():

        df_VectorAnalysis['number_of_vectors'] = np.where((df_VectorAnalysis['key']== key )
                                    , len(df_VectorAnalysis['Vect_1_Count'][(df_VectorAnalysis['key']==key)]),
                                    df_VectorAnalysis['number_of_vectors'])

        df_VectorAnalysis['max_1_Count'] = np.where((df_VectorAnalysis['key']== key )
                                    , max(df_VectorAnalysis['Vect_1_Count'][df_VectorAnalysis['key']==key]),
                                    df_VectorAnalysis['max_1_Count'])

        df_VectorAnalysis['min_1_Count'] = np.where((df_VectorAnalysis['key']== key )
                                    , min(df_VectorAnalysis['Vect_1_Count'][df_VectorAnalysis['key']==key]),
                                    df_VectorAnalysis['min_1_Count'])

        df_VectorAnalysis['average_1_Count'] = np.where((df_VectorAnalysis['key']== key )
                                    , np.mean(df_VectorAnalysis['Vect_1_Count'][df_VectorAnalysis['key']==key]),
                                    df_VectorAnalysis['average_1_Count'])

        df_VectorAnalysis['%10_quantile_1_Count'] = np.where((df_VectorAnalysis['key']== key )
                                    , df_VectorAnalysis['Vect_1_Count'][df_VectorAnalysis['key']==key].quantile(0.1),
                                    df_VectorAnalysis['%10_quantile_1_Count'])

        df_VectorAnalysis['%25_quantile_1_Count'] = np.where((df_VectorAnalysis['key']== key )
                                    , df_VectorAnalysis['Vect_1_Count'][df_VectorAnalysis['key']==key].quantile(0.25),
                                    df_VectorAnalysis['%25_quantile_1_Count'])

        df_VectorAnalysis['%30_quantile_1_Count'] = np.where((df_VectorAnalysis['key']== key )
                                    , df_VectorAnalysis['Vect_1_Count'][df_VectorAnalysis['key']==key].quantile(0.30),
                                    df_VectorAnalysis['%30_quantile_1_Count'])

        df_VectorAnalysis['median_quantile_1_Count'] = np.where((df_VectorAnalysis['key']== key )
                                    , df_VectorAnalysis['Vect_1_Count'][df_VectorAnalysis['key']==key].quantile(0.5),
                                    df_VectorAnalysis['median_quantile_1_Count'])


    df_SynthVectorAnalysis = pd.DataFrame(df_VectorAnalysis[['key','number_of_vectors','max_1_Count','min_1_Count','average_1_Count',
                                                        '%10_quantile_1_Count','%25_quantile_1_Count','%30_quantile_1_Count',
                                                        'median_quantile_1_Count']]).drop_duplicates(subset=None, keep='first', inplace=False)
    df_SynthVectorAnalysis = df_SynthVectorAnalysis.sort_values(['number_of_vectors'], ascending=[0])

    df_SynthVectorAnalysis = df_SynthVectorAnalysis.set_index(df_SynthVectorAnalysis['key'], drop=False)

    if argument == 'summary':
        return(df_SynthVectorAnalysis)
    elif argument == 'global':
        return(df_VectorAnalysis)
    else:
        print("bad argument : choose between 'summary' and 'global'")

def percent_of_total(myDataFrame,myColumnList):
    """
    This function goal is to convert each continuous columns of a determined
    list into a column were the values are the percentage of the sum of all
    columns included in the list.

    RESULT : The entry DataFrame with columns (included in 'myColumnList')
    converted into percentage of their sum.

    PARAMS :
    - 'myDataFrame' refers to the entry myDataFrame.
    - 'myColumnList' refers to the list  of columns with which we want to focus
    the analysis
    """
    myDataFrame['total'] = myDataFrame[myColumnList].sum(1)
    for column in myColumnList:
        myDataFrame[column] = 100*(myDataFrame[column]/ myDataFrame['total'])
    myDataFrame.drop('total',inplace=True,axis=1)
    return myDataFrame

def convert_in_list(myDataFrame,myColumn):
    from ast import literal_eval
    """
    This function goal is to convert a pandas column into a "list" datatype column
    IMPORTANT : The column values must match with the python lists pattern in order to be read and converted correctly.

    RESULT : The same column, with each value converted into an array : that's also possible to loop over the array values

    PARAMS :
    - myDataFrame : the entry DataFrame
    - myColumn : String, the column to convert
    """

    myDataFrame[myColumn] = myDataFrame[myColumn].apply(literal_eval)
    return myDataFrame

def develop(myDataFrame,myArrayColumn):
    import pandas as pd
    """
    This function goal is to develop the values contained in a pandas column which has list of values :
    IMPORTANT: The column 'myArrayColumn' must be read like a colum containing lists of values; if not, you should transform
    your column into a "list datatype column" thanks to the 'convert_in_list' function.

    RESULT : each value contained in a given list from a given row generates a new row : the other columns values are repeated

    PARAMS :
    - 'myDataFrame' : the entry DataFrame
    - 'myArrayColumn' : the list datatype column you want to generate one row per couple value/value_in_the_list
    """

    new_df = pd.DataFrame(columns = myDataFrame.columns)
    not_focus_cols = [x for x in myDataFrame.columns if x!=myArrayColumn]
    for idx in myDataFrame.index:
        focus_array = myDataFrame.loc[idx,myArrayColumn]
        old_values = myDataFrame.loc[idx,]

        for value in focus_array:
            dict_one_line = {}
            dict_one_line[myArrayColumn] = value
            idx_df = len(new_df)

            for col in not_focus_cols:
                dict_one_line[col] = old_values[col]
            new_df = new_df.append(dict_one_line,ignore_index=True)

    return new_df

def develop(myDataFrame,myArrayColumn):
    import pandas as pd
    """
    This function goal is to develop the values contained in a pandas column which has list of values :
    IMPORTANT: The column 'myArrayColumn' must be read like a colum containing lists of values; if not, you should transform
    your column into a "list datatype column" thanks to the 'convert_in_list' function.

    RESULT : each value contained in a given list from a given row generates a new row : the other columns values are repeated

    PARAMS :
    - 'myDataFrame' : the entry DataFrame
    - 'myArrayColumn' : the list datatype column you want to generate one row per couple value/value_in_the_list
    """
    new_df = pd.DataFrame(columns = myDataFrame.columns)
    not_focus_cols = [x for x in myDataFrame.columns if x!=myArrayColumn]
    for idx in myDataFrame.index:
        focus_array = myDataFrame.loc[idx,myArrayColumn]
        old_values = myDataFrame.loc[idx,]

        for value in focus_array:
            dict_one_line = {}
            dict_one_line[myArrayColumn] = value
            idx_df = len(new_df)

            for col in not_focus_cols:
                dict_one_line[col] = old_values[col]
            new_df = new_df.append(dict_one_line,ignore_index=True)

    return new_df

def group_by_frequency(myDataFrame,myColumn):
    import numpy as np
    """
    This function goal is to build an aggregated DataFrame which contains the occurences of the catagorical terms contained in
    'myColumn' args.

    RESULT : an aggregated DataFrame with the occurences of each values.
    - The DataFrame is sorted by descending occurences.
    - It also contains :
        - rank of each category in terms of occurences.
        - cumsum of occurences from the first value to the last one.
        - percent of total occurences covered by the upper categories at a given row.

    PARAMS :
    - 'myDataFrame' : the entry DataFrame
    - 'myColumn' : the column concerned by the frequencies count
    """
    grouped = myDataFrame.copy()
    grouped['occurences'] = 1
    grouped = grouped[[myColumn,'occurences']].groupby(myColumn).sum()
    grouped.sort_values(by='occurences', ascending=False, inplace=True)
    grouped['rank'] = range(1,len(grouped)+1)
    grouped['cumsum'] = np.cumsum(grouped['occurences'])
    grouped['percent_of_total'] = grouped['cumsum']/grouped['occurences'].sum()

    return grouped


def aggregate_col_in_list(myDataFrame,myColumnToAggregate,GroupByColumn,bool_replace_col):
    import pandas as pd
    """
    This function goal is to aggregate a given column values in order to get a list of it's values groupped by an other colum

    RESULT : The entry DataFrame :
    - with the same number of columns if the 'bool_replace_col' argument is 'True'
    - with one more column if the 'bool_replace_col' argument is 'False' : in this case the new colum has the name
    of 'myColumnToAggregate' with the suffix '_list'

    PARAMS :
    - 'MyDataFrame' |pandas.core.frame.DataFrame : The entry DataFrame
    - 'myColumnToAggregate'|str : The column you want to convert into a list
    - 'GroupByColumn' |str : The column you want to use in order to aggregate your 'myColumnToAggregate' column
    - 'bool_replace_col' |bool : A boolean argument with the value :
        - True if you want to replace the old column by the new one
        - False in the other case : the new colum has the name of 'myColumnToAggregate' with the suffix '_list'
    """
    temp_dataframe = pd.DataFrame(myDataFrame.groupby(GroupByColumn)[myColumnToAggregate].apply(list))
    temp_dataframe[GroupByColumn] = temp_dataframe.index
    myColumnToAggregate_new_name = myColumnToAggregate+"_list"

    if bool_replace_col == True:
        finalDataFrame = myDataFrame.drop(myColumnToAggregate,axis=1)
        finalDataFrame = pd.merge(finalDataFrame,temp_dataframe,how='left',on=GroupByColumn)

    else:
        temp_dataframe.rename(index=str, columns={myColumnToAggregate:myColumnToAggregate_new_name},inplace=True)
        finalDataFrame = pd.merge(myDataFrame,temp_dataframe,how='left',on=GroupByColumn)

    return finalDataFrame

def filter_list_of_list_values(myList,myFilterList):
    """
    This function goal is to filter values contained into a list of lists

    RESULT : A filtered list of lists

    PARAMS :
    - 'myList' : The list of lists you want to filter
    - 'myFilterList' : The list of values you want to delete into 'myList'

    EXAMPLE :
    L = [['Paris', 'Monaco', 'Washington', 'Lyon', 'Venise', 'Marseille'],
    ['New-York', 'NapleWashington', 'Turin', 'Chicago', 'Las Vegas'],
    ['Lyon', 'Rome', 'Chicago', 'Venise', 'Naple', 'Turin']]

    filter_list_of_list_values(L,['Lyon','Turin','Chicago'])
    >>  [['Paris', 'Monaco', 'Washington', 'Venise', 'Marseille'],
    >>  ['New-York', 'NapleWashington', 'Las Vegas'],
    >>  ['Rome', 'Venise', 'Naple']]
    """

    for index in range(len(myList)):
        sub_array = myList[index]
        for stopword in myFilterList :
            sub_array = list(filter(lambda a: a != stopword, sub_array))
        sub_array = [w for w in sub_array if not w in myFilterList]
        myList[index] = sub_array
    return myList

def dataframe_multiprocessing(myDataFrame,myFunction,myFunctionArg,NUMBER_OF_PROCS):
    import multiprocessing
    import pandas as pd
    import numpy as np
    """
    This function goal is to apply existing pandas transformation with multiprocessing.
    IMPORTANT : the transformations you use must be rows to rows transformation : this function is not adapted for aggregations
    so you need to use independant (in terms of rows) transformations

    RESULT : The result of your transformation ('myFunction' arg)

    PARAMS :
    - 'myDataFrame' : the entry DataFrame
    - 'myFunction' : the transformation/function you want to use with 'dataframe_multiprocessing'
    - 'myFunctionArg' : the argument of 'myFunction'
    - 'NUMBER_OF_PROCS' : the number of processors to use : the entry DataFrame will be divided into
    as many parts than processors used
    """

    working_DataFrame = myDataFrame.copy()
    output_schema = list(working_DataFrame.columns)
    working_DataFrame['old_index'] = working_DataFrame.index

    #We are going to execute a multiprocessing and split the DataFrame in as many parts than processors used :
    #DataFrame splitting :
    L_sub_dfs  = np.array_split(working_DataFrame, NUMBER_OF_PROCS)
    resultDataFrame = pd.DataFrame(columns=output_schema)

    print('Creating pool with %d processes\n' % NUMBER_OF_PROCS)
    with multiprocessing.Pool(NUMBER_OF_PROCS) as pool:
        # We initialize a list of tasks which each call the same function, with the same argument 'Tags', but
        #with a diffrent DataFrame
        TASKS = [(sub_df, myFunctionArg) for sub_df in L_sub_dfs]
        results = [pool.apply_async(develop, t) for t in TASKS]
        final_results = [r.get() for r in results]

        for sub_df_res in final_results:
            sub_df_res.index = sub_df_res['old_index']
            sub_df_res = sub_df_res[output_schema]
            resultDataFrame = resultDataFrame.append(sub_df_res)
            print("df_append")
    resultDataFrame.sort_index(inplace=True)

    del working_DataFrame
    return resultDataFrame

def class_my_files(myPath):
    """
    This function goal is to build a dictionnary of all the files available in
    a given repository, based on the files extensionss.

    RESULT : a dictionnary which maps all files to their extensions

    PARAMS :
    - 'myPath' : the path of the repository in which you want to map files.
    """
    from os import listdir
    import re

    L_files = listdir(myPath)
    dict_extensions = {}
    extensions = [r'.csv',r'.xls$',r'.xlsx',r'.json',r'.txt',r'.p']

    for ext in extensions :
        regex = re.compile(ext)
        selected_files = list(filter(regex.search, L_files))

        clean_ext = re.sub('\.|\$','',ext)
        dict_extensions[clean_ext] = selected_files

    return dict_extensions

def smart_lemmatize(myList,myStopwordsList):
    """
    This function goal is to lemmatize the words contained in a list ('myList' arg)

    RESULT : a list of lemmatized words
    see more at :
    https://textminingonline.com/dive-into-nltk-part-iv-stemming-and-lemmatization
    https://www.nltk.org/book/ch05.html
    https://www.kaggle.com/c/word2vec-nlp-tutorial#part-1-for-beginners-bag-of-words

    PARAMS :
    - 'myList' : the entry list of words
    - 'myStopwordsList' : The list you want to use if you don't want to lemmatize some words

    EXAMPLE :
    smart_lemmatize(['are','a','king','of','snakes','which','comes','from','the','thieves','countries'])
    >>>>>>>>>>>>>>> ['be','a','king','of','snake','which','come','from','the','thief','country']
    """
    from nltk import pos_tag
    from nltk.stem import WordNetLemmatizer

    lemmatized_list = []
    #We use 'pos_tag' ("part of speech tag) in order to regognize the grammatical essence of each word :
    grammatical_essence_list = pos_tag(myList,tagset='universal')
    #We create a dict in order to map the results of the args which comes from the 'pos_tag' processing
    dict_pos_tag_mapping = {'ADJ':'a','ADJ_SAT':'s','ADV':'r','NOUN':'n','VERB':'v'}
    smart_lemmatizer = WordNetLemmatizer()

    #And we associate a smart lemme to each word contained in our list :
    for tuple_ in grammatical_essence_list :
        word = tuple_[0]
        raw_essence = tuple_[1]
        if raw_essence in dict_pos_tag_mapping.keys():
            essence = dict_pos_tag_mapping[raw_essence]
        else:
            essence = 'n'
        if not word in myStopwordsList:
            lemme = smart_lemmatizer.lemmatize(word,essence)
            lemmatized_list.append(lemme)
        else:
            lemmatized_list.append(word)

    return lemmatized_list

def lemmatize_list_of_list(myListOfList,myLemmatizeFunction,myStopwordsList):
    for index in range(len(myListOfList)):
        target_list = myListOfList[index]
        myListOfList[index] = smart_lemmatize(target_list,myStopwordsList)
    return myListOfList

def n_grams_counters_list(myList_of_lists,N_GRAM):
    """
    This function goal is to build a list of Counter's objects, based on n-grams,
    and built into lists contained into a main list (the 'myList_of_lists' arg)
    IMPORTANT : each value contained in the main list MUST be a list itself
    TIP : apply this functions on a pandas DataFrame column which contains tokenized text

    RESULT : A list where every row contains a Counter object mapping n-grams and their occurences.

    PARAMS :
    - 'myList_of_lists' : The entry list which contains lists of values (a tokenized text for example)
    - 'N_GRAM' | integer : A number which specifies what type of n-gram you want to use
    """
    from nltk import ngrams
    from collections import Counter
    L_n_grams = []

    for index in range(len(myList_of_lists)):

        sub_array = myList_of_lists[index]
        n_gram_array = list(ngrams(sub_array,N_GRAM))
        counter = Counter(n_gram_array)
        L_n_grams.append(counter)

    return L_n_grams
