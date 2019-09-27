import numpy as np
import matplotlib.pyplot as plt
import os

shape_name = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
              'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar',
              'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant',
              'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet',
              'tv_stand', 'vase', 'wardrobe', 'xbox']

def load_data(file_path):
    items = os.listdir(file_path)
    filelist = []
    datas = []
    for name in items:
        if name.endswith(".npz"):
            filelist.append(name)
    filelist.sort(key=lambda x: int(x.split('_')[-1][:-4]))
    for name in filelist:
        data = np.load(file_path+name)
        datas.append(data)
    return datas,len(filelist)

def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def pseudo_color(gray):
    if gray < 0.25:
        r = 0.0
        g = 4*gray
        b = 1.0
    elif gray < 0.5:
        r = 0.0
        g = 1.0
        b = 2.0 - 4*gray
    elif gray <0.75:
        r = 4*gray-2.0
        g = 1.0
        b = 0
    else:
        r = 1.0
        g = 4.0 - 4*gray
        b = 0.0
    return (r,g,b)

# datas of each class
# def plot_lines_1(Ys,label):
#     Ys = np.concatenate(Ys).transpose()
#     shape = Ys.shape
#     print(shape)
#     line_num = shape[0]
#     point_num = shape[1]
#     Xs = [i for i in range(point_num)]
#     for i in range(line_num):
#         color = pseudo_color(i / line_num)
#         plt.plot(Xs,Ys[i],color=color,label=label)
#         plt.legend()
#
# # ROC
# def plot_lines_2(Ys,Xs):
#     line_num = len(Ys)
#     print(line_num)
#     for i in range(line_num):
#         color = pseudo_color(i / line_num)
#         plt.plot(Xs[i],Ys[i],color=color,label='epoch_'+str(2*i))
#         plt.legend()

def plot_line(y,x=None,label=None,color=None,axis_name=['x','y']):
    if x is None:
        x = range(len(y))
    plt.plot(x,y,color=color,label=label)
    plt.legend()

def plot_multi_lines(Xs,Ys,labels):
    x_num = len(Xs)
    y_num = len(Ys)
    l_num = len(labels)
    assert x_num == y_num
    assert l_num == y_num or l_num == 1
    for i in range(y_num):
        color = pseudo_color(i / y_num)
        label = labels[0] + '_'+str(i) if l_num==1 else labels[i]
        plot_line(Ys[i],Xs[i],color=color,label=label)

if __name__ == '__main__':
    # file_path = '/home/ym/桌面/remote_data/999/EvaluateData/'
    # file_path = '/home/ym/桌面/remote_data/compare/'
    file_path = '/home/ym/桌面/remote_data/pointCNN/EvalData/'
    datas,epoch = load_data(file_path)
    confusion_matrix,accuracy,precision,recall,f1,fpr,tpr,auc = [],[],[],[],[],[],[],[]
    for data in datas:
        confusion_matrix.append(data['confusion_matrix'])
        accuracy.append(data['accuracy'].item())
        auc.append(data['auc'].item())
        f1.append(data['f1'].item())
        precision.append(data['precision'])
        recall.append(data['recall'])
        fpr.append(data['fpr'])
        tpr.append(data['tpr'])

    # plot_confusion_matrix(confusion_matrix[24],shape_name,'confusion_matrix')
    plt.grid()
    # plot_line(auc,label='auc')
    # plot_multi_lines(fpr, tpr, ['model'])
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # plot_multi_lines(recall, precision, ['epoch'])
    # plt.xlabel('recall')
    # plt.ylabel('precision')
    plt.show()