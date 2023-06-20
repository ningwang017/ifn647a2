# Set working directory - all subsequent folder references relative to this folder
# user_wd = r'C:\Users\Guy\Desktop\IFN647 - Text, Web and Media Analytics\Assignment 2'

# -*- coding: utf-8 -*-
"""
IFN647 - Text, Web and Media Analytics
Assignment 2

N10610961: Dean Lay
N11148870: Ning Wang
To run:
    - Ensure variable "user_wd" is updated with the root path of subsequent files or folders
    - Add required files to the root folder
        "common-english-words.txt"
        "Queries.txt"
    - Create folders in the root folders:
        'Result' 
        'Output'
    - Extract the provided 'DataSets' and 'Feedback' data into the root folder
"""

#%% Data loading functions
def load_stopwords():
    '''
    Loads the typically provided file of stop words for query parsing.
    No inputs required.
    Returns a list of stop words.
    '''
    filepath_stopwords = os.getcwd() + r'\common-english-words.txt'
    
    with open(filepath_stopwords, 'r') as f:
        stopwordList = f.read().split(',')
        
    return stopwordList

def load_queries():
    '''
    Gets Query details from file and loads it into a DataFrame.
    No inputs required.
    Returns a DataFrame of queries.
    '''
    # Initialising
    filepath_queries = os.getcwd() + r'\Queries.txt'
    
    # Results lists
    datasets = []
    titles = []
    descs = []
    narrs = []
    
    # Text placeholders
    desc_txt = ''
    narr_txt = ''

    # Tag flags    
    flag_desc = 0
    flag_narr = 0
    
    with open(filepath_queries, 'r') as f:
        for line in f:
            ### Handle <desc>
            # Flag <desc> tag
            if line.startswith('<desc>'):
                flag_desc = 1
                continue
            # Reset desc flag and write desc to list
            if (flag_desc == 1) & (len(line) == 1):
                flag_desc = 0
                descs.append(desc_txt)
                desc_txt = ''
                continue
            # Extract desc text
            if flag_desc == 1:
                desc_txt += line.replace('\n', ' ')
                continue
    
            ### Handle <narr>
            # Flag <narr> tag
            if line.startswith('<narr>'):
                flag_narr = 1
                continue
            # Reset narr flag and write narr to list
            if (flag_narr == 1) & (len(line) == 1):
                flag_narr = 0
                narrs.append(narr_txt)
                narr_txt = ''
                continue
            # Extract <narr> text
            if flag_narr == 1:
                narr_txt += line.replace('\n', ' ')
                continue
            
            ### Query number and title
            if line.startswith('<num>'):
                datasets.append(line.strip("<num> Number: R").replace('\n', '').strip(' '))
                continue
            if line.startswith('<title>'):
                titles.append(line.strip("<title> ").replace('\n', ''))
                continue
        
    return pd.DataFrame({
        'dataset': datasets,
        'titles': titles,
        'descriptions': descs,
        'narratives': narrs
        })

def load_feedback():
    '''
    Loads all the feedback into a DataFrame.
    No inputs required.
    Returns a DataFrame of feedback information.
    '''
    # Initialising
    folderpath_feedback = os.getcwd() + r'\Feedback'    
    
    # Lists to save data
    list_topic = []
    list_docid = []
    list_rel = []
        
    # Load all files in the folder into a dataframe
    for file in os.listdir(folderpath_feedback):
        with open(os.path.join(folderpath_feedback, file), 'r') as f:
            # Read and split to lists
            all_rows = f.read()
            all_rows = all_rows.split('\n')
            for row in all_rows:
                try:
                    l = row.split()
                    list_topic.append(l[0])
                    list_docid.append(l[1])
                    list_rel.append(l[2])
                except:
                    pass # Handles blank rows at end of files
    
    # Combine to DataFrame
    return pd.DataFrame({
        'topic': list_topic,
        'docid': list_docid,
        'actual_rel': [int(val) for val in list_rel]
        })

#%% BM25 Baseline Model
#%%% Classes and functions
class DocWords:   
    # Contains information about a document.
    def __init__(self, docID, terms, doc_len):
        # Attributes
        self.docID = docID     # Document ID
        self.terms = terms     # Dictionary of (term: freq) pairs
        self.doc_len = doc_len # Words in document (before tokenizing/stemming/stop words)
        
    def addTerm(self, tf_dict):
        # Adds terms and their frequencies
        # If not pre-existing, creates new
        for key in tf_dict:
            try:
                self.terms[key] += tf_dict[key]
            except:
                self.terms[key] = tf_dict[key]
        
class BowColl:
    # A class to store and add DocWords objects
    # DocWords objects are stored as list items in the 'Coll' attribute
    def __init__(self, Coll, weights):
        self.Coll = Coll # List of DocWords
        self.weights = weights # List dicts containing of (term: weight) 
        
    def addDocWords(self, DocWords):
        #Adds new DocWords to the collection
        self.Coll.append(DocWords)
        
    def addWeights(self, weights):
        self.weights.append(weights)
        
    def getSummary(self):
        # Prints and returns string of docID, # terms and # words
        # E.g. "Document 741299 contains 96 indexing terms and have total 199 words"
        print_string = """"""
        for d in self.Coll:
            summaryInfo = f"Document {d.docID} contains {len(d.terms)} indexing terms and has a total of {d.doc_len} words\n"
            print_string += summaryInfo
            
            # Sort terms before printing
            dict_ordered = {k: v for k, v in sorted(d.terms.items(), key=lambda item: item[1], reverse=True)}
            for i in dict_ordered.items():
                print_string += i[0] + ': ' + str(i[1]) + '\n'
        
            print_string += '\n'
        
        print(print_string)
        return print_string        

def parse_docs(inputpath, stop_words, include_files=[]):
    # Takes an input XML file path and list of stop words
    # include_files is an optional parameter - only files in the list will be included
    # Returns a Bag of Words Collection, where each XML in the input path is a DocWord object
    
    # Initialise a BoW collection
    bow_coll = BowColl([], [])
           
    # Check DataSet match
    # Check docid match
    
    for file_path in os.listdir(inputpath):
        if len(include_files) == 0 or (file_path in include_files):
            # Initialise
            curr_doc = DocWords(docID = '', terms = {}, doc_len = 0)
            word_count = 0
            flag = 0
            line_doc = {}
        
            # Loop through each file, gather metadata
            f = open(os.path.join(inputpath, file_path), 'r')
            for line in f:    
                # Get Item ID if exists
                if 'itemid' in line:
                    bits = line.split()
                    for b in bits:
                        if b.startswith('itemid'):
                            # Strip all letters and punctuation
                            b = b\
                            .translate(str.maketrans('', '', string.ascii_letters))\
                            .translate(str.maketrans('', '', string.punctuation))
                            curr_doc.docID = b
                            break
    
                # Check if text starts - start flag if so
                if line.startswith('<text>'):
                    flag = 1
                    continue
                # Check if text ends - update flag and break
                if line.startswith('</text>'):
                    break
                # START PROCESSING IF IN TEXT BLOCK
                if flag == 1:
                    line_word_count, line_doc = parse_query(line, stop_words)
                    #Update word count and term-freq dictionary
                    word_count += line_word_count
                    curr_doc.addTerm(line_doc)
                    
            # Close file - No more lines to parse
            f.close()
            
            # Update final word count
            curr_doc.doc_len = word_count
            # Add to Collection
            bow_coll.addDocWords(curr_doc)
            del(curr_doc)
            
    return bow_coll

def parse_query(query0, stop_words):
    # Cleans and tokenizes input string "query0"
    # Returns:
    #   word_count (int): word count before tokenizing
    #   line_doc (dict): term, frequency key-value pairs
    
    # DEFINITIONS #
    # A "word" is defined as group of alphabetical characters greater than length 3
    # A "token" is a word that is not a stop word, and has been stemmed

    # Initialise doc of token information to return
    line_doc = {}
    
    # Specific text to remove (inc. tags)
    strip_list = ['<p>', '</p>', '&quot;']

    # Word length criteria
    min_word_len = 3
    
    # Remove trailing
    query0 = query0.strip()

    # Remove known characters
    for i in strip_list:
        query0 = query0.replace(i, '')
       
    # Remove digits and punctuation
    query0 = query0\
    .translate(str.maketrans('', '', string.digits))\
    .translate(str.maketrans('', '', string.punctuation))

    # Lower case
    query0 = query0.lower()
    
    # Split into terms and save unadulterated word count (after minimum length applied)
    words = query0.split()
    words = [i for i in words if len(i) >= min_word_len]
    word_count = len(words)

    # Remove stop words and stem
    tokens = [stem(i) for i in words if i not in stop_words]
    
    # Make a dictionary of token counts
    for token in tokens:
        try:
            line_doc[token] += 1
        except:
            line_doc[token] = 1
    
    return word_count, line_doc

def calc_df(coll, print_output = True):
    # Calculates document-frequency for a collection `coll` of DocWords
    # In this case, the input is a BowColl object
    # Optional parameter `print_output` controls console output
    # Returns a {term: df, ...} dictionary
    
    # Get counts
    doc_count = len(coll.Coll)
    term_count = sum([len(t.terms) for t in coll.Coll])
    all_terms = []
    [all_terms.extend(list(t.terms.keys())) for t in coll.Coll]
    unique_terms = len(set(all_terms))
    
    # Get document-frequency
    # Looping through each document
    Index = {}
    for doc in coll.Coll:
        for term in doc.terms.keys():
             # Use existing term in the index
            try:
                Index[term] += 1 #doc.terms[term]
            # Create new term in the index
            except KeyError:
                Index[term] = 1 #doc.terms[term]

    # Sort terms before printing
    if print_output == True:
        print(f"There are {doc_count} documents in this collection, with {term_count} terms ({unique_terms} unique).\nThe following are the terms' document-frequency:")
        print_string = """"""
        dict_ordered = {k: v for k, v in sorted(Index.items(), key=lambda item: item[1], reverse=True)}
        for i in dict_ordered.items():
            print_string += i[0] + ': ' + str(i[1]) + '\n'
        print(print_string)
    return Index

def avg_doc_len(coll):
    # Returns average length of all docs in a BoW collection
    total_len = 0
    num_docs = 0
    for bow in coll.Coll:
        num_docs += 1
        total_len += bow.doc_len
    return(total_len/num_docs)

def bm25(coll, q, df):
    # For a query `q`, returns a BM25 score per doc based on
    # a BoW collection `coll` and data frequency dict `df`
    res = {}
    
    # BoW Collection constant
    avdl = avg_doc_len(coll)
    N = len(coll.Coll) * 100000
    
    # Empirical Constants
    b = 0.75
    k1 = 1.2
    k2 = 500
    
    # Assume no relevance information
    R = 0
    ri = 0
    
    # Get BM25 score for each document
    for doc in coll.Coll:
        score_current = 0
        dl = doc.doc_len
        K = k1 * ((1-b) + b*(dl)/avdl)
            
        for term in q.keys():
            # Try to calculate BM25
            # Any missing terms means equation goes to 0, i.e. do not update
            try:    
                # Term counts
                ni = df[term]
                fi = doc.terms[term]
                qfi = q[term]

                # Equation sections ("_n" implies numerator)
                t1_n = (ri+0.5) / (R-ri+0.5)
                t1_d = (ni-ri+0.5) / (N-ni-R+ri+0.5)
                t2 = ((k1+1)*fi) / (K+fi)
                t3 = ((k2+1)*qfi) / (k2+qfi)

                score_current += np.log10((t1_n/t1_d) * t2 * t3)
            
            except KeyError:
                pass
        res[doc.docID] = score_current
        ordered_res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
        #print(f"{doc.docID}: {score_current}")
    return ordered_res

def run_bm25(top_n_print = 12):
    '''
    Executes BM25 for each query and dataset.
    Outputs:
        - .txt with top_n results into "Appendix_Output" folder
        - .dat with all document scores (ordered) into "Result" folder
    Parameters
    ----------
    top_n_print : int
        Number of top results to save into appendix text file. Default is 12.

    Returns
    -------
    None.

    '''
    print_string = '''***BM25 Baseline Model Results***\n'''    # String for printing results
    
    # Per query, calculate BM25 score in corresponding dataset
    for counter, dataset_num in enumerate(df_queries['dataset']):
        # Corresponding dataset folder
        dataset_path = os.path.join(folder_datasets, 'Dataset' + dataset_num)
    
        # Get Bag-of-words collection    
        coll_docs = parse_docs(dataset_path, stopwordList)
    
        # Get document frequencies in collection
        df = calc_df(coll_docs, print_output = False)
        
        # Calculate BM25 based on descriptions
        current_score = bm25(coll_docs, parse_query(df_queries['descriptions'][counter], stopwordList)[1], df)
        
        # Print top-n
        top_keys = []
        top_scores = []
        print_string += f"Query{dataset_num} (docID | Weight):\n"
        for k in list(current_score.keys())[0:top_n_print]:
            top_keys.append(k)
            top_scores.append(current_score[k])
            print_string += f"{k} {current_score[k]}\n"
        print_string += '\n'
        print(print_string)
        
        # Save top-n to file
        filename_appendix_output = 'Baseline_top' + str(top_n_print) + '.txt'
        with open(os.getcwd() + '\\Output\\' + filename_appendix_output, 'w') as f:
            f.write(print_string)
        
        # Save all
        res_df = pd.DataFrame({
            'keys': [k for k in current_score.keys()],
            'scores': [current_score[k] for k in current_score.keys()]
            })
        
        # Write all to file
        filename_output = 'Baseline_R' + dataset_num + 'Ranking.dat'
        res_df.to_csv(os.getcwd() + '\\Result\\' +  filename_output, sep = ' ', header = False, index = False)
    
#%% My_model1

import math

def tfidf(coll, df, ndocs):
    """
    Calculates tf*idf weight for every term in a document
    
    Args:
        doc (dict): a DocWords object or a dictionary of {term:freq, ...}
        df (dict): a {term:df, ...} dictionary
        ndocs (int): the number of documents in a given DocWords collection
    
    Returns:
        dict: a {term:tfidf_weight, ...} dictionary for the given document doc
    """
    tfidf_dict = {}

    for term in coll.Coll:

        for x in term.terms:
            tf = 1 + math.log(term.terms[x])
            idf = math.log(ndocs / df[x])
            score = tf * idf

            tfidf_dict[term.docID] = score

    ordered_res = {k: v for k, v in sorted(tfidf_dict.items(), key=lambda item: item[1], reverse=True)}
    return ordered_res



def run_tfidf_model(top_n = 12):
    print_string = '''***IR-based Model Results***\n'''    # String for printing results
    
    # Per query, calculate BM25 score in corresponding dataset
    for counter, dataset_num in enumerate(df_queries['dataset']):
        # Corresponding dataset folder
        dataset_path = os.path.join(folder_datasets, 'Dataset' + dataset_num)
    
        # Get Bag-of-words collection    
        coll_docs = parse_docs(dataset_path, stopwordList)

        print(len(coll_docs.Coll))

        # Get document frequencies in collection
        df = calc_df(coll_docs, print_output = False)
  
        current_score = tfidf(coll_docs, df, len(coll_docs.Coll))
        
        # Print top-n
        top_keys = []
        top_scores = []
        print_string += f"Query{dataset_num} (docID | Weight):\n"
        for k in list(current_score.keys())[0:top_n]:
            top_keys.append(k)
            top_scores.append(current_score[k])
            print_string += f"{k} {current_score[k]}\n"
        print_string += '\n'
        print(print_string)
        
        # Save top-n to file
        filename_appendix_output = 'tfidf_top' + str(top_n) + '.txt'
        with open(os.getcwd() + '\\Output\\' + filename_appendix_output, 'w') as f:
            f.write(print_string)
        
        # Save all
        res_df = pd.DataFrame({
            'keys': [k for k in current_score.keys()],
            'scores': [current_score[k] for k in current_score.keys()]
            })
        
        # Write all to file
        filename_output = 'tfidf_R' + dataset_num + 'Ranking.dat'
        res_df.to_csv(os.getcwd() + '\\Result\\' +  filename_output, sep = ' ', header = False, index = False)
     
        



#%% My_model2: Pseudo-Relevance Feedback
def run_pseudo_rel(top_n_docs=5, top_n_terms = 10, top_n_save = 12):
    '''
    Performs Query Expansion based on initial baseline BM25 model results.
    Therefore requires that results from the original BM25 model exist in the
    `Result` folder.
    Outputs:
        - .txt with top_n_save results into "Output" folder
        - .dat with all document scores (ordered) into "Result" folder
    Parameters
    ----------
    top_n_docs : int
        Number of results from original BM25 results to consider as "relevant". Default is 5
    top_n_terms : int
        Number of top terms to add to query expansion. Default is 10.
    top_n_save : int
        Number of top results to save into appendix text file. Default is 12.
    
    Returns
    -------
    None.

    '''
    # Load the previously saved BM25 baseline model results
    # Keep only the number of documents as provided by `top_n_rel`
    # Lists to save data
    l_dataset = []
    l_docid = []
    
    for filename in os.listdir(folder_result):
        # Limit loading to Baseline results only
        if filename.startswith('Baseline'):
            with open(os.path.join(folder_result, filename), 'r') as f:
                all_rows = f.read()
                all_rows = all_rows.split('\n')
                
                count = 0 # Tracking results returned
                for row in all_rows:
                    if count < top_n_docs:
                        if row != '': # Handles any blank rows
                            l = row.split(' ')
                            l_docid.append(l[0])
                            
                            # Metadata from filename
                            file_text = filename.replace('Ranking.dat', '')
                            l_dataset.append(file_text.split('_')[1].replace('R', ''))
                            
                            # Increment
                            count += 1
                    else: 
                        break
    
    df_bm25 = pd.DataFrame({
        'dataset': l_dataset,
        'docid': l_docid,
        'file': [i+'.xml' for i in l_docid]
        })  
    
    # Per query, calculate BM25 score in corresponding dataset
    print_string = '''***My_model2 Pseudo Relevance Results***\n'''    # String for printing results
    for counter, dataset_num in enumerate(df_queries['dataset']):
        # Corresponding dataset folder
        dataset_path = os.path.join(folder_datasets, 'Dataset' + dataset_num)
    
        #### Query Expansion
        # Get Bag-of-words collection for top files
        include_files = df_bm25[df_bm25['dataset'] == dataset_num]['file'].to_list()
        coll_docs = parse_docs(dataset_path, stopwordList, include_files)
         
        # Get document frequencies in top collection
        df = calc_df(coll_docs, print_output = False)
        dict_ordered = {k: v for k, v in sorted(df.items(), key=lambda item: item[1], reverse=True)}
        top_terms = []
        for k in list(dict_ordered.keys())[0:top_n_terms]:
            top_terms.append(k)
        
        # Load original query and add top terms
        query = df_queries['descriptions'][counter]
        for t in top_terms:
            query += ' ' + t

        #### Run BM25 as normal, but with expanded query
        coll_docs = parse_docs(dataset_path, stopwordList)
        df = calc_df(coll_docs, print_output = False)
        current_score = bm25(coll_docs, parse_query(query, stopwordList)[1], df)

        # Print top-n
        top_keys = []
        top_scores = []
        print_string += f"Query{dataset_num} (docID | Weight):\n"
        for k in list(current_score.keys())[0:top_n_save]:
            top_keys.append(k)
            top_scores.append(current_score[k])
            print_string += f"{k} {current_score[k]}\n"
        print_string += '\n'
        print(print_string)
        
        # Save top results to file
        filename_appendix_output = 'My_model2_top' + str(top_n_save) + '.txt'
        with open(os.getcwd() + '\\Output\\' + filename_appendix_output, 'w') as f:
            f.write(print_string)
    
        # Save all
        res_df = pd.DataFrame({
            'keys': [k for k in current_score.keys()],
            'scores': [current_score[k] for k in current_score.keys()]
            })
        
        # Write all to file
        filename_output = 'My_model2_R' + dataset_num + 'Ranking.dat'
        res_df.to_csv(os.getcwd() + '\\Result\\' +  filename_output, sep = ' ', header = False, index = False)

    return


#%% Task 4: Evaluation
# Load model results into a dataframe
def load_model_results():
    '''
    Loads all model results into a DataFrame.
    Adds relevance column based on if result in `top_n` and its weighting.
    Weightings of 0 cannot be relevant.
    Assumes result files are pre-ordered in weight descending order.

    Returns
    -------
    df_res : TYPE
        DataFrame of all model results with relevance column.

    '''
    
    # Lists to save data
    l_model = []
    l_topic = []
    l_docid = []
    l_weight = []
    
    for filename in os.listdir(folder_result):
        with open(os.path.join(folder_result, filename), 'r') as f:
            all_rows = f.read()
            all_rows = all_rows.split('\n')
            for row in all_rows:
                if row != '': # Handles any blank rows
                    l = row.split(' ')
                    l_docid.append(l[0])
                    l_weight.append(l[1])
                    
                    # Metadata from filename
                    file_text = filename.replace('Ranking.dat', '')
                    l_model.append(file_text.rsplit('_',1)[0])
                    l_topic.append(file_text.rsplit('_',1)[1])
    
    df_res = pd.DataFrame({
        'model': l_model,
        'topic': l_topic,
        'docid': l_docid,
        'weight': [float(val) for val in l_weight]
        })
    
    # Join on the actual relevance from feedback file, useful for evaluation
    df_res = df_res.merge(df_feedback, how = 'left', left_on = ['topic', 'docid'], right_on = ['topic', 'docid'])

    return df_res

# Determine effectiveness measures 
def calc_precision(df):
    # Calculate Precision per model/topic combination
    l_prec = []
    l_pos = []
    for m in df['model'].unique():
        for t in df['topic'].unique():
            # Get relevant documents from feedback
            Dpos = df_feedback[(df_feedback['topic'] == t) & (df_feedback['actual_rel'] == 1)]['docid'].to_list()
        
            res_subset = df[(df['model'] == m) & (df['topic'] == t)]
            
            # Counter for relevant documents
            rel_count = 0
            
            # Calculate precision/position for each document in the subset
            for i in range(len(res_subset)):
                position = i+1
                
                if res_subset['docid'].iloc[i] in Dpos:
                    rel_count +=1 
                    
                l_prec.append(rel_count/position)
                l_pos.append(position)
                
    df = df.assign(precision = l_prec,
                   position = l_pos)
    
    return df


def evaluate_performance(df, at_pos):
   
    '''
    Evaluates the performance of different model results as provided in `df`,
    using 3 different performance metrics:
        - Average Performance
        - Mean Average Performance @ Position `at_pos`
        - DCG @ Position `at_pos`
    Generates graphical summaries of the results and saves them in the Output
    folder.
    Returns DataFrames required for t-tests.

    Parameters
    ----------
    df : DataFrame
        Contains pre-loaded model results as passed in from `load_model_results`
        and `calc_precision` functions.
    at_pos : int
        Position to calculate MAP and cumulative DCG at.

    Returns
    -------
    df_ap_g1 : DataFrame
        Average Precision values per model and topic .
    df_p_g1 : DataFrame
        Precision at `at_pos` per model and topic.
    df_dcg : DataFrame
        Cumlative DCG at `at_pos` per model and topic

    '''
    sns.set_theme()
   # sns.set_context(font_scale = 1.5)
    sns.set(rc={'figure.figsize':(10,4),
                'figure.labelsize': 'x-large',
                'legend.fontsize': 'medium',
                'axes.labelsize': 'medium',
                'axes.titlesize':'medium',
                'xtick.labelsize':'medium',
                'ytick.labelsize':'medium'
                })
    
    #### Average Precision
    # Graph 1: Boxplot of Precision distribution per model/topic
    df_ap_g1 = df.groupby(['model', 'topic'])['precision'].mean().reset_index()
    ap_g1 = sns.boxplot(data = df_ap_g1, x = 'model', y='precision', hue = 'model', dodge=False)
    ap_g1 = sns.stripplot(data= df_ap_g1, x = 'model', y = 'precision', jitter = 0.025, color = 'k', alpha = 0.3)
    ap_g1.set(title='Average Precision of Topics',
              ylabel='Average Precision',
              ylim = (-.01,1.05)
              )
    ap_g1.get_legend().remove()
    fig = ap_g1.get_figure()
    fig.savefig(folder_output + '\\1_1_AveragePrecision_Boxplot')
    plt.clf()
    
    # Graph 2: Bar plot of Mean Average Precision per model
    df_ap_g2 = df_ap_g1.groupby('model')['precision'].mean().reset_index()
    ap_g2 = sns.barplot(data = df_ap_g2, x = 'model', y='precision', hue = 'model', dodge=False)
    ap_g2.set(title='Mean Average Precision',
              ylabel='Average Precision',
              ylim = (-.01,1.05)
              )
    ap_g2.get_legend().remove()
    fig = ap_g2.get_figure()
    # Add data labels to bar
    for i in ap_g2.containers:
        ap_g2.bar_label(i,)
    fig.savefig(folder_output + '\\1_2MeanAveragePrecision_Barplot')
    plt.clf()
    
    #### Precision@at_pos
    # Graph 1: Boxplot of Precision @ at_pos
    df_p_g1 = df[df['position']==at_pos]
    p_g1 = sns.boxplot(data = df_p_g1, x = 'model', y='precision', hue = 'model', dodge=False)
    ap_g1 = sns.stripplot(data= df_p_g1, x = 'model', y = 'precision', jitter = 0.025, color = 'k', alpha = 0.3)
    p_g1.set(title='Precision at Position ' + str(at_pos),
             ylabel='Precision', 
             ylim = (-.01,1.05)
             )
    p_g1.get_legend().remove()
    fig = p_g1.get_figure()
    fig.savefig(folder_output + '\\2_1_PrecisionAtPosition' + str(at_pos) + '_Boxplot')
    plt.clf()
    
    # Graph 2: Bar plot of Average Precision @ at_pos
    df_p_g2 = df_p_g1.groupby('model')['precision'].mean().reset_index()
    ap_g2 = sns.barplot(data = df_p_g2, x = 'model', y='precision', hue = 'model', dodge=False)
    ap_g2.set(title='Average Precision at Position ' + str(at_pos),
              ylabel='Average Precision',
              ylim = (-.01,1.05)
              )
    ap_g2.get_legend().remove()
    # Add data labels to bar
    for i in ap_g2.containers:
        ap_g2.bar_label(i,)
    fig = ap_g2.get_figure()
    fig.savefig(folder_output + '\\2_2_AveragePositionAt' + str(at_pos) + '_Barplot')
    plt.clf()

    #### DCG@at_pos
    # Calculate DCG@at_pos
    # Reduce set to relevant records
    df_subset = df[df['position']<=at_pos]
    
    l_topic = []
    l_model = []
    l_dcg = []
    
    for m in df_subset['model'].unique():
        for t in df_subset['topic'].unique():
            # Reduce set down to specific model and topic
            df_current = df_subset[(df_subset['model'] == m) & (df_subset['topic'] == t)].reset_index()
            rel1 = df_current['actual_rel'].iloc[0]
            dcg = rel1
            
            for i in range(1, len(df_current)):
                dcg += df_current['actual_rel'].iloc[i] / np.log2(i+1+1)
                
            l_topic.append(t)
            l_model.append(m)
            l_dcg.append(dcg)

    df_dcg = pd.DataFrame({
        'model': l_model,
        'topic': l_topic,
        'dcg': l_dcg
        })
    
    # Calculate maximum theoretical Cumulative DCG
    max_dcg = 1
    for i in range(2, at_pos+1):
        max_dcg += (1/np.log2(i+1))
    
    # Graph 1: Boxplot of DCG @ at_pos
    dcg_g1 = sns.boxplot(data = df_dcg, x = 'model', y='dcg', hue = 'model', dodge=False)
    dcg_g1 = sns.stripplot(data= df_dcg, x = 'model', y = 'dcg', jitter = 0.025, color = 'k', alpha = 0.3)
    dcg_g1.axhline(y = max_dcg, color = 'red', linestyle = '--', linewidth = 1)
    dcg_g1.annotate('Max Theoretical DCG: ' + str(round(max_dcg, 3)), 
                    xy = (-0.48, max_dcg + 0.1)
                    )
    dcg_g1.set(title='Cumulative DCG at Position ' + str(at_pos),
               ylabel='Cumulative DCG',
               ylim = (0,max_dcg+0.5)
              )
    dcg_g1.get_legend().remove()
    
    fig = dcg_g1.get_figure()
    
    fig.savefig(folder_output + '\\3_1_CumulativeDCGAtPosition' + str(at_pos) + '_Boxplot')
    plt.clf()
    
    # Graph 2: Barplot of Average DCG @ at_pos
    dcg_g2 = sns.barplot(data = df_dcg.groupby(['model'])['dcg'].mean().reset_index(), x = 'model', y='dcg', hue = 'model', dodge=False)
    dcg_g2.axhline(y = max_dcg, color = 'red', linestyle = '--', linewidth = 1)
    dcg_g2.annotate('Max Theoretical DCG: ' + str(round(max_dcg, 3)), 
                    xy = (-0.48, max_dcg + 0.1)
                    )
    dcg_g2.set(title='Average Cumulative DCG at Position ' + str(at_pos),
               ylabel='Average Cumulative DCG',
               ylim = (0,max_dcg+0.5)
              )
    dcg_g2.get_legend().remove()
    # Add data labels to bar
    for i in dcg_g2.containers:
        dcg_g2.bar_label(i,)
    fig = dcg_g2.get_figure()
    fig.savefig(folder_output + '\\3_2_AverageCumulativeDCGAtPosition' + str(at_pos) + '_Barplot')
    plt.clf()    
    
    return df_ap_g1, df_p_g1, df_dcg



#%% Task 5: Recommend best model


def generate_t_test(output1, output2):

    """
    compare two outputs from two different models using t-test formula
    return t_statistic and p_value
    
    """

    t_statistic, p_value = stats.ttest_ind(output1, output2, alternative='two-sided')

    return t_statistic, p_value

def run_t_test(df):

    if 'precision' in df.columns:  #if using precision values
        baseline = df['precision'].tolist()[:50]
        model_2 = df['precision'].tolist()[50:100]
        model_1 = df['precision'].tolist()[100:]
    elif 'dcg' in df.columns: #if using dcg values 
        baseline = df['dcg'].tolist()[:50]
        model_2 = df['dcg'].tolist()[50:100]
        model_1 = df['dcg'].tolist()[100:]

    # print 3 outputs in terminal
    baseline_model1 = generate_t_test(baseline, model_1)
    print("\nComparing Baseline model and TF-IDF model ")
    print("T-Statistic:", baseline_model1[0])
    print("P-Value:", baseline_model1[1])

    print("\nComparing Baseline model and Pseudo-Relevance model ")
    baseline_model2 = generate_t_test(baseline, model_2)
    print("T-Statistic:", baseline_model2[0])
    print("P-Value:", baseline_model2[1])

    print("\nComparing TF-IDF model and Pseudo-Relevance model ")
    model1_model2 = generate_t_test(model_1, model_2)
    print("T-Statistic:", model1_model2[0])
    print("P-Value:", model1_model2[1])

    return


#%% Main code block
# Load libraries
import os
import string
from stemming.porter2 import stem
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Set working directory - all subsequent folder references relative to this folder
curr_path = os.getcwd()
os.chdir(curr_path)

# Folders
folder_datasets = os.getcwd() + r"\DataSets"
folder_feedback = os.getcwd() + r'\Feedback'
folder_result = os.getcwd() + r'\Result'
folder_output = os.getcwd() + r'\Output'

# Load common data
stopwordList = load_stopwords()
df_queries = load_queries()
df_feedback = load_feedback()    

# Implement Baseline model: BM25
#run_bm25()

# Implement My_model1
#run_tfidf_model()

# Implement My_model2: Pseudo-Relevance
#run_pseudo_rel(top_n_docs = 5, top_n_terms = 10, top_n_save = 12)

# Evaluate results
df_res = load_model_results()
df_res = calc_precision(df_res)
a,b,c = evaluate_performance(df_res, at_pos = 12)

print(c)

print("<----- Significance Test on Average Precision Values ----->")
run_t_test(a)

print("<----- Significance Test on Precision Values ----->")
run_t_test(b)

print("<----- Significance Test on DCG Values ----->")
run_t_test(c)
