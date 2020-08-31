from scipy import stats
import csv
from iksimilarity import IKSimilarityTools as IKST
from iksimilarityw2v import IKWord2VecTools as IKW2V
import os

""" This script will run through 1 or more saved models and evaluate their performance.
    To do so, the code makes use of a few publically available datasets that provide multiple word pairs 
    along with average rankings of similarity/relatedness from human subjects. For each such dataset, the 
    Spearman correlation coefficient is computed by comparing the human scores for each pair to the scores
    given by the model for each pair. The datasets contain different pairs.

    To evaluate model(s), simply run this script. When run, it will list out the names of all models saved 
    on disk. It then prompts you, where you can input one model name or a few of the given model names
    separated by commas. 
        e.g:
        my_model
        my_first_model, my_second_model
    
    Since this script was written to test performance of a number of different models, it is capable of testing both
    fastText and Word2Vec models. As such, it makes some assumptions: if a model name contains 'w2v' (in some form), 
    it will treat the model as a Word2Vec model, otherwise it will treat it as fastText. 

    To be tested, the model must have vectors saved in the models/keyed_vectors directory. If the models 
    being tested were created through the iksimilarity module, this is where models are stored.

    Output
    -----------
    A csv file (for each model) with the correlation coeff. for each dataset used for comparison, saved in the
    datasets directory.

    Note that Spearman coefficients are only one way to evaluate such models, they are not certain measures. 
"""

def main():
    print("Current models available for evaluation:\n")
    for folder in os.listdir('models/keyed_vectors'):
        if not folder.startswith('.'):
            print(folder)
    models = input("\nInput one of the above model names, or multiple names separated by commas. For all, enter 'a': ").split(',')
    if models[0] == 'a':
        models = os.listdir('models/keyed_vectors')
    for model_name in models:
        if model_name.startswith('.'): # If using all models, ignore hidden files in the dir
            continue
        model_name = model_name.strip()
        if 'w2v' in model_name.lower():
            test_tool = IKW2V(pmodel_name=model_name)
        else:
            test_tool = IKST(pmodel_name=model_name)
        ws353 = test_dataset(test_tool, 'ws353')
        ws353_sim = test_dataset(test_tool, 'ws353_s')
        ws353_rel = test_dataset(test_tool, 'ws353_r')
        rw = test_dataset(test_tool, 'rw')
        men = test_dataset(test_tool, 'men')
        mturk771 = test_dataset(test_tool, 'mturk771')
        simlex999 = test_dataset(test_tool, 'simlex999')
        outputfile = 'datasets/testoutput{}.csv'.format(model_name)
        with open(outputfile, 'a+', newline='') as output:
            writer = csv.writer(output, delimiter=',')
            rows = [['Dataset', 'Spearman', 'OOV Ratio'],['WordSim 353', ws353[0], ws353[1]],['WordSim 353 Similarity', ws353_sim[0], ws353_sim[1]],
            ['WordSim 353 Relatedness', ws353_rel[0], ws353_rel[1]], ['Rare Words', rw[0], rw[1]], ['MEN', men[0], men[1]], ['MTURK-771', mturk771[0], mturk771[1]], 
            ['SimLex-999', simlex999[0], simlex999[1]]]
            writer.writerows(rows)
        print("Evaluation of model %s complete, see datasets folder for output." % model_name)

def test_dataset(test_tool, dataset):
    if dataset == 'ws353_s':
        word_pairs = 'datasets/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt'
        delim = '\t'
    elif dataset == 'ws353_r': 
        word_pairs = 'datasets/wordsim353_sim_rel/wordsim_relatedness_goldstandard.txt'
        delim = '\t'
    elif dataset == 'ws353':
        word_pairs = 'datasets/wordsim353_sim_rel/wordsim353_agreed.txt'
        delim = '\t'
    elif dataset == 'rw':
        word_pairs = 'datasets/rw/rw.txt'
        delim = '\t'
    elif dataset == 'men':
        word_pairs = 'datasets/MEN/MEN_dataset_natural_form_full'
        delim = ' '
    elif dataset == 'mturk771':
        word_pairs = 'datasets/MTURK-771.csv'
        delim = ','
    elif dataset == 'simlex999':
        word_pairs = 'datasets/SimLex-999/SimLex-999.txt'
        delim = '\t'

    result = test_tool.evaluate_word_pairs(word_pairs=word_pairs, delimiter=delim, case_insensitive=True)

    return (result[0], result[1])


if __name__ == "__main__":
    main()
