# Intro
This my project created for the primary goal to learn basic use of PyTorch and its subsidiary libraries.
This CGI vs AI classifier utilizes a 3D Convolutional Neural Network to classify whether a chosen video was created with CGI or if it was AI-generated.

I trained with all available Sora AI clips, animated CGI clips, and movie clips from popular movies such as Pirates of the Carribean, Jurassic Park, and Iron Man.
My dataset and checkpoints are NOT currently in this Git Repo, as including those would have made the size of this repository over 60GB. 
My dataset and checkpoints will be available at this Sharepoint link: https://pitt-my.sharepoint.com/:f:/g/personal/dzz4_pitt_edu/Eouggte1C9hPqbyBzQnLHP4BXj2JDiOzND6XwjOPq-kX0Q?e=jXHMTm

# Results 
    My results are not representative of the peak capabilities of this model. I don't have a good source for creating a dataset - therefore I have to manually clip and classify videos - nor do I have the necessary 
    hardware to train at a reasonable speed. After 1019 epochs [training loops], I achieved a 75% testing accuracy on unseen data, which is a 15% increase after 473 epochs from the last testing session at 546 epochs. I believe this is a great success considering I haven't trained the model to its full potential, as by the time I stopped, the loss value was still dropping per epoch - meaning that it was still traversing towards a minimum value on the gradient descent.

# To-do 
Apply PCA dimensionality-reduction and expand training dataset.

Usage:
    
    [1]: Place your own training data, or my training dataset into the 'data' directory. The c0 folder should hold AI-generated videos and c1 for CGI. 
        Run "process_and_augment('data')", which will create two .npy files: 'VideoData.npy' containing the raw 
        video data and 'VideoLabels.npy', which contains the labels pertaining to the 'VideoData.npy' in order by index. 
        This process additionally generates a copy of each video, with augments applied to it - this was to compensate for my lack of training data'. 
        Also create your testing dataset in the same fashion, but instead of in the 'data' directory, place it in the
        'TestingData' directory. The testing data will not be used at all until step [3]. 
        Preprocess your testing data by running the function "process_no_augment('TestingData')" - this will create a similar
        'VideoData.npy' and 'VideoLabels.npy' in the "TestingData" directory; without any additional augmentations so that 
        the videos being tested are only the ones you chose to input.
    
    [2]: Ensure that on line 183 of "Model.py", the video_data_module.setup(stage='HERE') has parameters 'train' to configure the trainer to 
        train, validate, and test only on the training dataset, as we haven't created a testing dataset yet. Additionally, 
        choose the number of epochs (training loops) you want to run the training program for on line 206 in the Trainer() class.       
    
    [3]: Test your program by ensuring that the max_epochs parameter on line 206 of Model.py matches the last epoch you ended at, 
        then changing the line 183 stage parameter to 'test'
