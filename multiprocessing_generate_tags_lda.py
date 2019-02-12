import multiprocessing
import pickle
import numpy as np
import pandas as pd
import time

from context import temp_files_path
print("OK")
from tailored_functions import generate_tags_lda, generate_tags_in_list_lda, find_similar_neighbors
print("OK")
execution_params = pickle.load(open(temp_files_path+"execution_params.p", "rb" ))

N_PROCS = execution_params['N_PROCS']

list_of_corpus = execution_params['list_of_corpus']
lda_word_sim_matrix = execution_params['lda_word_sim_matrix']
lda_tag_sim_matrix = execution_params['lda_tag_sim_matrix']
existing_tags = execution_params['existing_tags']
NEIGHBORS = execution_params['NEIGHBORS']
recommendation_filter = execution_params['recommendation_filter']
QUANTILE_THRESHOLD = execution_params['QUANTILE_THRESHOLD']



if __name__ == '__main__':
    print("Multiprocessing")
    #We are going to execute a multiprocessing and split the list in as many parts than processors used :
    #DataFrame splitting :
    L_sub_lists_of_corpus  = np.array_split(list_of_corpus, N_PROCS)
    L_sub_lists_of_tags  = np.array_split(recommendation_filter, N_PROCS)
    final_List = []
    start_time = time.time()

    print('Creating pool with %d processes\n' % N_PROCS)
    with multiprocessing.Pool(N_PROCS) as pool:
        # We initialize a list of tasks which each call the same function, but
        #with a diffrent list
        TASKS = [(corpus, lda_word_sim_matrix, lda_tag_sim_matrix, existing_tags, NEIGHBORS, tag_list, QUANTILE_THRESHOLD) \
        for corpus,tag_list in zip(L_sub_lists_of_corpus,L_sub_lists_of_tags)]

        print("TASK ready")
        results = [pool.apply_async(generate_tags_in_list_lda, t) for t in TASKS]
        print("results ready")
        final_results = [r.get() for r in results]
        print("final_results ready")
        print("appending sub lists :")

        for sub_list_res in final_results:
            if len(sub_list_res)>0:
                final_List+=list(sub_list_res)
        print("success")

    print("--- %s seconds ---" % (time.time() - start_time))
    print('the result file is available at : \n temp_files_path+"L_recommended_tags.p"')
    pickle.dump(final_List,open(temp_files_path+"L_recommended_tags.p", "wb"))
