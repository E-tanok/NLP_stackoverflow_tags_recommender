import multiprocessing
import pickle
import numpy as np
import time

from context import temp_files_path

from functions import smart_lemmatize, lemmatize_list_of_list

execution_params = pickle.load(open(temp_files_path+"execution_params.p", "rb" ))
N_PROCS = execution_params['N_PROCS']
tokenized_text = execution_params['tokenized_text']
stopwords = execution_params['stopwords']

if __name__ == '__main__':
    print("Multiprocessing")
    #We are going to execute a multiprocessing and split the list in as many parts than processors used :
    #DataFrame splitting :
    L_sub_lists  = np.array_split(tokenized_text, N_PROCS)
    final_List = []
    start_time = time.time()

    print('Creating pool with %d processes\n' % N_PROCS)
    with multiprocessing.Pool(N_PROCS) as pool:
        # We initialize a list of tasks which each call the same function, but
        #with a diffrent list
        TASKS = [(sub_list,smart_lemmatize,stopwords) for sub_list in L_sub_lists]
        print("TASK ready")
        results = [pool.apply_async(lemmatize_list_of_list, t) for t in TASKS]
        print("results ready")
        final_results = [r.get() for r in results]
        print("final_results ready")
        print("appending sub lists :")

        for sub_list_res in final_results:
            if len(sub_list_res)>0:
                final_List+=list(sub_list_res)
        print("success")

    print("--- %s seconds ---" % (time.time() - start_time))
    print('the result file is available at : \n temp_files_path+"tokenized_text_lemmatized.p"')
    pickle.dump(final_List,open(temp_files_path+"tokenized_text_lemmatized.p", "wb"))
