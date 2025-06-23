import csv
import os
import numpy as np
import openpyxl
data_folder="/media/pjc/expriment/mdd_exam/pjc/EV_GCN-MDD/data/ABIDE_pcp/cpac/filt_noglobal/"
data_folder2="/media/pjc/expriment/mdd_exam/pjc/EV_GCN-MDD/data/Phenotypic_V1_0b_preprocessed1 (1035).csv"
subject_IDs = np.genfromtxt(os.path.join(data_folder, 'subject_IDs.txt'), dtype=str)
list2=[]

with open(data_folder2) as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        # if row['ID'] in subject_list:
        #     scores_dict[row['ID']] = row[score]
        list2.append(row['SUB_ID'])

a = 0
for id in list2:
    if id not in subject_IDs:
        csv_file.delete_rows(id,1)
        a += 1
        print(id)

print(a)