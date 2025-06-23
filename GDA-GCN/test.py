import numpy as np

prediction= [[0.324083 , 0.675917],
 [0.02630129, 0.97369869],
 [0.18709901, 0.81290098],
 [0.30669567, 0.69330435],
 [0.33875624, 0.66124376],
 [0.30583398, 0.69416601],
 [0.79597417, 0.20402583],
 [0.43065772, 0.5693423 ],
 [0.19529595, 0.80470404],
 [0.13520707, 0.86479294]]
# all_test=1
# k=222
# fname = '/media/pjc/expriment/Work2/Data_aug_Model/for_test/result.txt'
# f = open(fname, 'a',encoding='utf-8')
#
# f.write('test=%d\n'%all_test)
# f.write('k==%d\n'%k)
# f.close()
train_id=[1,4,5]
k=1
# print(prediction[4])
prediction=np.array(prediction)
for i in train_id:
    # print(prediction[i])
    prediction[i]=prediction[i]+k
    # print(prediction)
print(prediction)