import numpy as np
import random
import json

def target_function(x):
    return -2*np.arctan(x-5)+np.pi

def add_noise(x,y,noise=1, samples=5, seed=4):
    y_new=[]
    for i,y_value in enumerate(y):

        np.random.seed(seed=int(x[i]*y[i]*1000*seed) )
        y_new.append(y[i]+np.random.normal(0, 0.8, samples,)[0])
    return y_new


def get_data(samples_per_x=1000,seed=4,exp=2,samples_from_each=2000, grid=0.01):
    random.seed(seed)
    samples=samples_per_x


    x_values=np.arange(0.0, 10.0, grid)

    np.random.seed(seed=4)
    y_values=np.random.randint(0, 2*np.pi*100000, int(samples_per_x*10/grid)) / 100000
    x_values=np.array([[x for i in range(samples)] for x in x_values])
    x_values=x_values.reshape(len(x_values)*samples)
    data=np.array([x_values,y_values,x_values]).T

    scale=lambda x: add_noise(x,target_function(x),samples=1)

    data[:,2]=data[:,1]>scale(data[:,0])
    data= data[data[:,1].argsort()]
    # client2_data=data[data[:,0]<6]

    # x^5
    p1 = np.array([(exp ** x) for x in x_values])
    # p=np.array([max(0.000000001,np.arctan(x-8)+1) for x in x_values])
    p1 = p1 / p1.sum()

    p2 = np.array([(exp ** (10 - x))  for x in x_values])
    # p=np.array([max(0.000000001,np.arctan(x-8)+1) for x in x_values])
    p2 = p2 / p2.sum()
    client1_data_final = data[np.random.choice(data.shape[0], samples_from_each, p=p1, replace=False), :]
    client2_data_final = data[np.random.choice(data.shape[0],samples_from_each, p=p2, replace=False), :]

    data=data[np.random.choice(data.shape[0], samples_from_each*2, replace=False),:]
    # data=data[labels]
    y=np.array([target_function(x) for x in np.arange(0.0, 10.0, grid)])
    y=np.array([np.arange(0.0, 10.0, grid),y]).T

    return data, client1_data_final,client2_data_final,y


def save_data_as_json(client1_data_final,client2_data_final,data,seed=4):
    random.seed(seed)
    client1_data_final_json=[json.dumps({"x":row[0],"x2":row[1],"label":row[2]})+'\n' for row in client1_data_final]
    client2_data_final_json=[json.dumps({"x":row[0],"x2":row[1],"label":row[2]})+'\n' for row in client2_data_final]
    random.shuffle(client1_data_final)
    random.shuffle(client2_data_final)
    data_json=[json.dumps({"x":row[0],"x2":row[1],"label":row[2]})+'\n' for row in data]

    with open(f'datasets/train_r0.jsonl', 'w+') as f:
        f.writelines(client1_data_final_json)
    with open(f'datasets/train_r1.jsonl', 'w+') as f:
        f.writelines(client2_data_final_json)
    with open(f'datasets/test_r0.jsonl', 'w+') as f:
        f.writelines(data_json)
    with open(f'datasets/test_r1.jsonl', 'w+') as f:
        f.writelines(data_json)
    with open(f'datasets/validation_r1.jsonl', 'w+') as f:
        f.writelines(data_json)
    with open(f'datasets/validation_r0.jsonl', 'w+') as f:
        f.writelines(data_json)



from IPython import get_ipython
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except:
    pass
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plot_data(datas):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])

    for data in datas:
        ax.scatter(x=data[:, 0], y=data[:, 1], s=0.4)

    fig.show()