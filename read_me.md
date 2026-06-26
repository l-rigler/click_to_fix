!!! work with pyton 3.10 !!!
you also need a nvidia GPU to run it (might work also on CPU but slowly)

To install the trainer in the nnu-net folder: 
1 - activate your env 
2 - run install.sh 

To run the inferer go in the root of the repo and type -> python -m scripts.gui_launcher

!!! format requirement for the inference!!!!!
1 - in the gui_launcher the model folder path needs to go to the nnUNet_results folder of your model 
2 - in your input_folder every image should have the nnU-net format : IMAGE_COMMON_NAME ID CHANNEL
    2.1 - the image should be the channel 0000
    2.2 - clicks channels are next 0001 -> nbr of label

To run a training : 
1 - make sure that your data are in a nnUNet_raw folder & respect the nnU-net data format 
2 - like in inference : 
    2.1 - the image should be the channel 0000
    2.2 - clicks channels are next 0001 -> nbr of label
3 - use the nnUNetv2_plan_and_preprocess command
4 - just launch the nnUNetv2_train commande while specifiying the interactive trainer with --tr trainer_cubic_weighted_loss