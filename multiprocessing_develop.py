import multiprocessing
import pickle
import numpy as np
import pandas as pd
import time

from context import temp_files_path

from functions import develop

execution_params = pickle.load(open(temp_files_path+"execution_params.p", "rb" ))
N_PROCS = execution_params['N_PROCS']
dataframe = execution_params['dataframe']
column_to_develop = execution_params['column_to_develop']
developped_dataframe = pd.DataFrame(columns=(list(dataframe.columns)))

if __name__ == '__main__':
    print("Multiprocessing")
    #We are going to execute a multiprocessing and split the list in as many parts than processors used :
    #DataFrame splitting :
    L_sub_dfs  = np.array_split(dataframe, N_PROCS)
    start_time = time.time()

    print('Creating pool with %d processes\n' % N_PROCS)
    with multiprocessing.Pool(N_PROCS) as pool:
        # We initialize a list of tasks which each call the same function, but
        #with a diffrent DataFrame
        TASKS = [(sub_df, column_to_develop) for sub_df in L_sub_dfs]

        results = [pool.apply_async(develop, t) for t in TASKS]
        final_results = [r.get() for r in results]

    for sub_df_res in final_results:
        sub_df_res.index = range(len(sub_df_res))
        developped_dataframe = developped_dataframe.append(sub_df_res)
        print("df_append")

    developped_dataframe.sort_index(inplace=True)

    print("--- %s seconds ---" % (time.time() - start_time))
    pickle.dump(developped_dataframe,open(temp_files_path+"developped_dataframe.p", "wb"))
    print('the result file is available at : \n temp_files_path+"developped_dataframe.p"')
