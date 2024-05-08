#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/print_results.py
#                                                                             
# PROGRAMMER: 
# DATE CREATED:
# REVISED DATE: 
# PURPOSE: Create a function print_results that prints the results statistics
#          from the results statistics dictionary (results_stats_dic). It 
#          should also allow the user to be able to print out cases of misclassified
#          dogs and cases of misclassified breeds of dog using the Results 
#          dictionary (results_dic).  
#         This function inputs:
#            -The results dictionary as results_dic within print_results 
#             function and results for the function call within main.
#            -The results statistics dictionary as results_stats_dic within 
#             print_results function and results_stats for the function call within main.
#            -The CNN model architecture as model wihtin print_results function
#             and in_arg.arch for the function call within main. 
#            -Prints Incorrectly Classified Dogs as print_incorrect_dogs within
#             print_results function and set as either boolean value True or 
#             False in the function call within main (defaults to False)
#            -Prints Incorrectly Classified Breeds as print_incorrect_breed within
#             print_results function and set as either boolean value True or 
#             False in the function call within main (defaults to False)
#         This function does not output anything other than printing a summary
#         of the final results.
##
import numpy as np

def format_pet_classifier_table(pet_header, pet_values, classifier_header, classifier_values):
        """
        Formats pet and classifier results into a string of rows with aligned columns.
        Parameters:
                pet_header - string label constant for pet_values
                pet_values - array of strings with pet names
                classifier_header - string label constant for classifier_values
                classifier_values - array of strings with classifier results
        Returns:
                string - formatted tabel with input parameters aligned vertically
        """ 
        pet_headers = np.full(len(pet_values), pet_header)
        classifier_headers = np.full(len(classifier_values), classifier_header)
        max_pet_label_size = max(map(lambda p: len(p), pet_values))
        max_classifier_size = max(map(lambda p: len(p), classifier_values))
        rows = zip(pet_headers, pet_values, classifier_headers, classifier_values)

        column_widths = [len(pet_header), max_pet_label_size, len(classifier_header), max_classifier_size]
        column_alignments = ['<', '>', '<', '>']

        formatted_rows = []
        for row in rows:
                this_row = ''
                for column, width, alignment in zip(row, column_widths, column_alignments):
                       this_row += " {0:{align}{width}} ".format(column, align=alignment, width=width)
                formatted_rows.append(this_row)

        return "\n".join(formatted_rows)


def print_results(results_dic, results_stats_dic, model, 
                  print_incorrect_dogs = False, print_incorrect_breed = False):
    """
    Prints summary results on the classification and then prints incorrectly 
    classified dogs and incorrectly classified dog breeds if user indicates 
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and 
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 
                            0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 
                            'as-a' dog and 0 = Classifier classifies image  
                            'as-NOT-a' dog.
      results_stats_dic - Dictionary that contains the results statistics (either
                   a  percentage or a count) where the key is the statistic's 
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value 
      model - Indicates which CNN model architecture will be used by the 
              classifier function to classify the pet images,
              values must be either: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and 
                             False doesn't print anything(default) (bool)  
      print_incorrect_breed - True prints incorrectly classified dog breeds and 
                              False doesn't print anything(default) (bool) 
    Returns:
           None - simply printing results.
    """
    # list the stats we want to pring
    stat_labels = ['n_images', 'n_dogs_img', 'n_notdogs_img', 'pct_match', 'pct_correct_dogs', 'pct_correct_breed', 'pct_correct_notdogs']

    print(f"CNN Model: {model}")
    for label in stat_labels:
        # print each label, adding a "%" char to the end when label starts with 'p'
        print(f"{label}: {results_stats_dic[label]}{'%' if label[0] == 'p' else ''}")


    # updated conditional check per review guidance and from hints file
    # it passes the check if the correct + incorrect dog predictions are not equal
    # to the over all image count, since if everything was correctly predicted
    # these would sum to the same value.
    if (print_incorrect_dogs and 
        ( (results_stats_dic['n_correct_dogs'] + results_stats_dic['n_correct_notdogs'])
          != results_stats_dic['n_images'] ) 
       ):
        # print when the classifier and the actual label values disagree
        print('Incorrect Dog/NOT Dog Assignments: ')
        pet_values = [record[0] for record in results_dic.values() if record[4] != record[3]]
        classifier_values = [record[1] for record in results_dic.values() if record[4] != record[3]]
        formatted_rows = format_pet_classifier_table('Pet Label:', pet_values, 'Classifier:', classifier_values)
        print(formatted_rows)


    # updated conditional check per review guidance and hints file
    if (print_incorrect_breed and 
        (results_stats_dic['n_correct_dogs'] != results_stats_dic['n_correct_breed']) 
       ):
        # if it's a dog, and the classifier knows it's a dog, but it guessed the wrong breed
        print('Incorrect Dog Breed Assignments: ')
        pet_values = [record[0] for record in results_dic.values() if not record[2] and record[4] and record[3]]
        classifier_values = [record[1] for record in results_dic.values() if not record[2] and record[4] and record[3]]
        formatted_rows = format_pet_classifier_table('Pet Label:', pet_values, 'Classifier:', classifier_values)
        print(formatted_rows)
                
