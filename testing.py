import numpy as np 
import pandas as pd 
import cv2
import random
from pathlib import Path
from keras import models
from keras.models import model_from_json
from keras_preprocessing import image
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, accuracy_score
import os
ds_dir = r"C:\Users\omk00\OneDrive\Desktop\EYESPY\ocular-disease-recognition-odir5k\preprocessed_images"
image_size = 64
dataset = []
labels = []
df = pd.read_csv("Full_df.csv")
df.head()

def has_disease(section):
    if "cataract" in section:
        return 1
    elif "age" in section:
        return 2
    elif "glau" in section:
        return 4
    elif "non" and "retinopathy" or "diabetic" and "retinopathy" in section:
        return 3
    else:
        return 0

df["left_disease"] = df["Left-Diagnostic Keywords"].apply(lambda x: has_disease(x))
df["right_disease"] = df["Right-Diagnostic Keywords"].apply(lambda x: has_disease(x))

left_cataract = df.loc[(df.C ==1) & (df.left_disease == 1)]["Left-Fundus"].values
left_retinopathy = df.loc[(df.D ==1) & (df.left_disease == 3)]["Left-Fundus"].sample(300, random_state=13).values
left_degeneration = df.loc[(df.A ==1) & (df.left_disease == 2)]["Left-Fundus"].values
left_glau = df.loc[(df.G ==1) & (df.left_disease == 4)]["Left-Fundus"].values
left_normal = df.loc[(df.C ==0) & (df["Left-Diagnostic Keywords"] == "normal fundus")]["Left-Fundus"].sample(300,random_state=13).values
right_cataract = df.loc[(df.C ==1) & (df.right_disease == 1)]["Right-Fundus"].values
right_retinopathy = df.loc[(df.D ==1) & (df.right_disease == 3)]["Right-Fundus"].sample(300, random_state=13).values
right_degeneration = df.loc[(df.A ==1) & (df.right_disease == 2)]["Right-Fundus"].values
right_glau = df.loc[(df.G ==1) & (df.left_disease == 4)]["Right-Fundus"].values
right_normal = df.loc[(df.C ==0) & (df["Right-Diagnostic Keywords"] == "normal fundus")]["Right-Fundus"].sample(300,random_state=13).values

print("Number of images in left C: {}".format(len(left_cataract)))
print("Number of images in right C: {}".format(len(right_cataract)))
print("Number of images in left D: {}".format(len(left_retinopathy)))
print("Number of images in right D: {}".format(len(right_retinopathy)))
print("Number of images in left A: {}".format(len(left_degeneration)))
print("Number of images in right A: {}".format(len(right_degeneration)))
print("Number of images in left G: {}".format(len(left_glau)))
print("Number of images in right G: {}".format(len(right_glau)))
print("Number of images in right N: {}".format(len(left_normal)))
print("Number of images in right N: {}".format(len(right_normal)))


cataract = np.concatenate((left_cataract,right_cataract),axis=0)
retinopathy = np.concatenate((left_retinopathy,right_retinopathy),axis=0)
glaucoma = np.concatenate((left_glau,right_glau),axis=0)
degeneration = np.concatenate((left_degeneration,right_degeneration),axis=0)
normal = np.concatenate((left_normal,right_normal),axis=0)

print(len(cataract),len(retinopathy), len(glaucoma), len(degeneration), len(normal))

def create_dataset(category, label):
    for img in tqdm(category):
        image_path = os.path.join(ds_dir, img)
        try:
            image = cv2.imread(image_path,cv2.IMREAD_COLOR)
            image = cv2.resize(image,(image_size, image_size))
        except:
            continue
        dataset.append([np.array(image),np.array(label)])
    random.shuffle(dataset)
    return dataset

dataset = create_dataset(cataract, 1)
dataset = create_dataset(degeneration, 2)
dataset = create_dataset(retinopathy, 3)
dataset = create_dataset(glaucoma, 4)
dataset = create_dataset(normal, 0)

len(dataset)

#features
x = np.array([i[0] for i in dataset]).reshape(-1,image_size,image_size,3)
#targets
y = np.array([i[1] for i in dataset])

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2)

y_train = to_categorical(y_train, 5)
y_test = to_categorical(y_test, 5)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

class_labels = [
    "1",
    "2",
    "3",
    "4",
    "0"
]

def class_prob_name(class_label):
    if int(class_label) == 1:
        return "cataract"
    elif int(class_label) == 2:
        return "macular degeneration"
    elif int(class_label) == 3:
        return "diabetic retinopathy"
    elif int(class_label) == 4:
        return "glaucoma"
    elif int(class_label) == 0:
        return "normal"

lb = LabelBinarizer()
f = Path("!_model_structure.json")
model_structure = f.read_text()
loaded_model = model_from_json(model_structure)
loaded_model.load_weights("!_saved_weights.h5")

y_pred = loaded_model.predict(x_test, batch_size=32, verbose=1)
o = y_pred
y_pred = (y_pred > 0.5)
s = y_pred
rounded_labels = np.argmax(y_pred, axis=1)
rounded_labels = lb.fit_transform(rounded_labels)
print(y_pred)
#one-third testing--------------------------
print(classification_report(rounded_labels,y_pred))
print(accuracy_score(rounded_labels, y_pred))
#two-third testing--------------------------
cnf_matrix = confusion_matrix(y_test.argmax(axis = 1), y_pred.argmax(axis = 1))
cmap = sns.cm.rocket_r
sns.heatmap(cnf_matrix, annot=True, fmt= 'g', cmap=cmap)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
#three-third testing------------------------
ovo_macro_auc = roc_auc_score(y_test, y_pred, multi_class="ovo", average="macro")
ovo_weighted_auc = roc_auc_score(y_test, y_pred, multi_class="ovo", average="weighted")
ovr_macro_auc = roc_auc_score(y_test, y_pred, multi_class="ovr", average="macro")
ovr_weighted_auc = roc_auc_score(y_test, y_pred, multi_class="ovr", average="weighted")

print("OvO ROC AUC scores:\n{:.5f} (macro),\n{:.5f} (weighted)".format(ovo_macro_auc, ovo_weighted_auc))
print("OvR ROC AUC scores:\n{:.5f} (macro),\n{:.5f} (weighted)".format(ovr_macro_auc, ovr_weighted_auc))