import os
import imageio
import cv2
import glob
import numpy as np
import random
import vidaug.augmentors as va
import threading #For future use incase more hardware optimizations are needed.
import pandas as pd
from npy_append_array import NpyAppendArray
import time
import shutil

#MARK: loaddata() function
def loaddata(Video_dir,n_classes):
    files = os.listdir(Video_dir)
    X = []
    labels = []
    
    for i in range(n_classes):
        path = os.path.join(Video_dir, 'c'+str(i),'*.mp4')
        print(path)
        files = glob.glob(path)
        for filename in files:
            labels.append(i)
            #Decreased from 200 to 50 in attempt to reduce CUDA VRAM overallocation
            X.append(load_video(filename, max_frames=50))
            #Preventing RAM overuse by appending to .npy files and reading.
            #Since loaddata is only going to be used with original, unaugmented set, can overwrite. 
            save_to_npy(np.array(X), np.array([i]), Video_dir)
            X = []
            labels = []
        
    Xout = np.load(Video_dir+'/VideoData.npy', mmap_mode="r+")
    Yout = np.load(Video_dir+'/VideoLabels.npy', mmap_mode="r+")
    return Xout , Yout

#MARK: load_video() function
def load_video(path, max_frames=0, resize=(300, 300)):
    print('Processing: {}'.format(os.path.basename(path)))
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames) / 255.0

#MARK: crop_center_square()
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

#MARK: to_gif()
def to_gif(images, lab):
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    i = 0
    prefix = ''

    if lab == 0:
        prefix = 'AI'
    else:
        prefix = 'CGI'
    
    while os.path.exists("data/AugmentedGifs/{}Augmented{}.gif".format(prefix,i)):
        i+=1
    imageio.mimsave("data/AugmentedGifs/{}Augmented{}.gif".format(prefix,i), converted_images,  fps=25)
    print("GIF made")


#MARK: augmentation()
def augmentation(X_train,Y_train, Folder='data'):
    #vidaug library provides functions for randomly applying video augments:
    sometimes = lambda aug: va.Sometimes(0.5, aug) 
    seq = va.Sequential([ 
        sometimes(va.RandomRotate(degrees=random.randrange(90))),
        sometimes(va.VerticalFlip()),
        sometimes(va.HorizontalFlip()),
        sometimes(va.GaussianBlur(1.5))
    ])
    X_train_aug = []
    Y_train_aug = []
    '''
    Making an augmented copy of each video as I have a very restricted dataset to train on.
    '''
    for i in range(len(X_train)):
        vid = X_train[i]
        video_aug = np.array(seq(vid))
        X_train_aug.append(video_aug)
        Y_train_aug.append(Y_train[i])
        save_to_npy(np.array(X_train_aug), np.array(Y_train_aug), Folder)
        #to_gif(video_aug, Y_train[i])
        X_train_aug = []
        Y_train_aug = []
    #Don't need output since I am saving all the data. 
    #return np.array(X_train_aug),np.array(Y_train_aug)

#MARK: save_to_npy
def save_to_npy(Xdata, Ydata, Folder='data'):
    data_path = Folder+'/VideoData.npy'
    label_path= Folder+ '/VideoLabels.npy'
    with NpyAppendArray(data_path, delete_if_exists=False) as npaa:
        npaa.append(Xdata)
    with NpyAppendArray(label_path, delete_if_exists=False) as npab:
        npab.append(Ydata)


#For processing training data, and subsequently creating more training data by augmenting.
def process_and_augment(Folder):
    if os.path.exists(Folder + '/VideoData.npy'):
        os.remove(Folder + '/VideoData.npy')
    if os.path.exists(Folder + '/VideoLabels.npy'):
        os.remove(Folder + '/VideoLabels.npy')

    Xin, Yin = loaddata(Folder, 2)
    #save_to_npy(Xin,Yin, Folder)
    Xin = np.load(Folder + '/VideoData.npy', mmap_mode='r+')
    Yin = np.load(Folder + '/VideoLabels.npy', mmap_mode='r+')
    augmentation(Xin, Yin, Folder)
    display(Folder)

def process_no_augment(Folder):
    #Function for staging videos for Testing, no augments.
    if os.path.exists(Folder + '/VideoData.npy'):
        os.remove(Folder + '/VideoData.npy')
    if os.path.exists(Folder + '/VideoLabels.npy'):
        os.remove(Folder + '/VideoLabels.npy')
    Xin, Yin = loaddata(Folder, 2)
    #save_to_npy(Xin,Yin, Folder) #Already embedded in "loaddata"
    Xin = np.load(Folder + '/VideoLabels.npy', mmap_mode='r+')
    Yin = np.load(Folder + '/VideoLabels.npy', mmap_mode='r+')
    display(Folder)

def display(Folder):
    Xin = np.load(Folder+'/VideoData.npy', mmap_mode='r+')
    Yin = np.load(Folder+'/VideoLabels.npy', mmap_mode='r+')
    for i in range(len(Xin)):
        print('Labeled {}, X[{}]: {}'.format(Yin[i], i, Xin[i].shape))

#Function(s) for moving the testing videos to training, once done with testing. 
def move_and_rename_files(src_dir, dest_dir):
    #List of all files in folder
    src_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

    #Count num files in the destination directory
    existing_files_count = len([name for name in os.listdir(dest_dir) if os.path.isfile(os.path.join(dest_dir, name))])

    #Move and rename from source to desination
    for i, filename in enumerate(src_files, start=1):
        src_path = os.path.join(src_dir, filename)
        new_filename = f"video{existing_files_count + i}.mp4"
        dest_path = os.path.join(dest_dir, new_filename)
        shutil.move(src_path, dest_path)
        print(f"Moved and renamed {filename} to {new_filename}")

def mv_test_to_train():
    src_dir_c0 = 'TestingData/c0'
    dest_dir_c0 = 'data/c0'
    src_dir_c1 = 'TestingData/c1'
    dest_dir_c1 = 'data/c1'
    #move and rename for both classes
    move_and_rename_files(src_dir_c0, dest_dir_c0)
    move_and_rename_files(src_dir_c1, dest_dir_c1)


if __name__=="__main__":
    process_no_augment('TestingData')
    #mv_test_to_train()
    #process_and_augment('data')
    