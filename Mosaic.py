
# coding: utf-8

# In[1]:


import numpy as np
from PIL import Image
import os


# In[2]:


import cv2


# In[2]:


def unpickle(file):
    import pickle
    with open(file, 'rb')as fo:
        dict = pickle.load(fo, encoding = 'latin1')
    return dict


# In[3]:


#print(dict.keys())
# dict_keys(['filenames','batch_label', 'fine_labels', 'coarse_labels', 'data'])


# In[4]:


def loadCifarBatch(filename, foldername):
    dict = unpickle(filename)
    batch_label = dict['batch_label']
    fine_labels = dict['fine_labels']
    coarse_labels = dict['coarse_labels']
    batch_label = np.array(batch_label)
    fine_labels = np.array(fine_labels)
    coarse_labels =np.array(coarse_labels)
    data = dict['data']
    if foldername == "train":
        data = data.reshape(50000, 3, 32, 32)
    elif foldername == "test":
            data = data.reshape(10000, 3, 32, 32)
    return data, batch_label, fine_labels, coarse_labels


# In[5]:


# name = classname + "_" +str(trainImgNum) + ".png" 
# saveimgname = os.path.join(foldername,classname, name)


# In[6]:


def saveImage(filename, foldername):
    imagedata, batch_label, fine_labels, coarse_labels = loadCifarBatch(filename, foldername)
    label2classdict = unpickle('cifar-100-python/meta')
    #print(label2classdict.keys())
    label2classfine = label2classdict['fine_label_names']
    trainImgNum = 0
    for i in range(imagedata.shape[0]):
        #print(i)
        image = imagedata[i]
        imgR, imgG, imgB = image[0], image[1], image[2]
        imgR = Image.fromarray(imgR)
        imgG = Image.fromarray(imgG)
        imgB = Image.fromarray(imgB) 
        img = Image.merge("RGB", (imgR, imgG, imgB))
        label = fine_labels[i]
        classname = label2classfine[label]
        if foldername == "train":
            name = str(trainImgNum) + ".jpg"
        elif foldername == "test":
            name = str(trainImgNum) + ".jpg"
        saveimgname = os.path.join("train", name)
        print(saveimgname)
        savepath = foldername
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        img.save(saveimgname, "JPEG")
        trainImgNum += 1


# In[7]:


def main():
    filename = 'cifar-100-python/train'
    foldername = "train"
    saveImage(filename, foldername)
if __name__ == "__main__":
    main()



# In[1]:


import cv2
img = cv2.imread("hh.jpg")
print(img.shape)
res = cv2.resize(img, (2400, 1800))
cv2.imwrite("hh_1.jpg", res)


# In[10]:


import photomosaic as pm

image = pm.imread("hh_1.jpg")
pool = pm.make_pool('train/*.jpg', sample_size=5000)
mosaic = pm.basic_mosaic(image, pool, (480, 360))
pm.imsave('montage2.jpg', mosaic)

