import matplotlib.pyplot as plt
import numpy as np
import read_data

shape_name=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
            'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box','guitar',
            'keyboard','lamp','laptop','mantel','monitor','night_stand','person','piano','plant',
            'radio','range_hood','sink','sofa','stairs','stool','table','tent','toilet',
            'tv_stand','vase','wardrobe','xbox']

inputTrain, trainLabel, inputTest, testLabel = read_data.load_data(2048, 'farthest_sampling','/home/ym/PycharmProjects/Fundamentals_of_Python/PointClassfication/data/')
trainLabel = np.concatenate([value for value in trainLabel.values()]).tolist()
testLabel = np.concatenate([value for value in testLabel.values()]).tolist()
len_trainLabel = len(trainLabel)
len_testLabel = len(testLabel)
num_test = [0]*40
num_train = [0]*40
for i in range(40):
    num_test[i] = testLabel.count(i)/len_testLabel
    num_train[i] = trainLabel.count(i)/len_trainLabel
print(num_test)
print(num_train)


index = np.arange(40)
bar_width = 0.4
plt.figure()
plt.bar(index, num_test, width=0.4, color=(0.8, 0.6, 0.1),label='evaluate dataset')
plt.bar(index + bar_width, num_train, width=0.4, color=(0.1, 0.8, 0.6),label = 'train dataset')
plt.xticks(index,shape_name,size='small',color=(0.6, 0.1, 0.8),fontsize=20,rotation=70)
for a,b in zip(index,num_test):
    plt.text(a, b+0.001, '%.3f' % b, ha='center', va= 'bottom',fontsize=15,rotation=90)
for a,b in zip(index+bar_width,num_train):
    plt.text(a, b+0.001, '%.3f' % b, ha='center', va= 'bottom',fontsize=15,rotation=90)
legend = plt.legend(fontsize = 20)
plt.xlabel('Classification',fontsize=20)
plt.ylabel('Rate',fontsize=20)
plt.show()

