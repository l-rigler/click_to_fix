import os
import matplotlib.pyplot as plt
from gui import my_io as io
import pathlib
from scipy.ndimage import gaussian_filter
from torch.nn.functional import pad 
import nnunetv2.inference.predict_from_raw_data as infer
import torch
import numpy as np 
from torch._dynamo import OptimizedModule


def make_gaussian(tile_size,point,sigma_scale = 1. / 8,
                     value_scaling_factor = 1, dtype=torch.float16, device=torch.device('cuda', 0)) \
        -> torch.Tensor:
    
    tmp = np.zeros(tile_size)
    coords = point
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

    gaussian_importance_map /= (torch.max(gaussian_importance_map) / value_scaling_factor)
    gaussian_importance_map = gaussian_importance_map.to(device=device, dtype=dtype)
    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    mask = gaussian_importance_map == 0
    gaussian_importance_map[mask] = torch.min(gaussian_importance_map[~mask])
    return gaussian_importance_map

class inferer():
    
    def __init__(self,model_training_output_dir,fold = (0,1,2,3,4),checkpoint_name='checkpoint_final'):
        self.model_training_output_dir = model_training_output_dir
        self.fold = fold
        self.checkpoint_name  = checkpoint_name
        

    def model_init(self,):

        os.environ['nnUNet_raw']= self.model_training_output_dir+'/nnUNet_raw'
        os.environ['nnUNet_preprocessed']= self.model_training_output_dir+'/nnUNet_preprocessed'
        os.environ['nnUNet_results']= self.model_training_output_dir+'/nnUNet_results'
        print(os.environ.get('nnUNet_raw'))
        print(os.environ.get('nnUNet_preprocessed'))
        print(os.environ.get('nnUNet_results'))
        self.predictor = infer.nnUNetPredictor(tile_step_size=0.5,
                            use_gaussian=True,
                            use_mirroring=True,
                            perform_everything_on_device=True,
                            device=torch.device('cuda:0'),
                            verbose=False,
                            verbose_preprocessing=False,
                            allow_tqdm=True)
        
        self.predictor.initialize_from_trained_model_folder(
        self.model_training_output_dir,
        self.fold,
        checkpoint_name= self.checkpoint_name+'.pth'
        )
        
        #maybe get sliding window size here 
        self.win_d,self.win_h,self.win_l = self.predictor.configuration_manager.patch_size
        return "model initalized"

    def first_prediction(self,input):
        
        self.logits = self.predictor.predict_logits_from_preprocessed_data(input)
        self.logits = self.logits.to(torch.device('cuda:0'))
        self.proba = torch.softmax(self.logits,dim = 0)
        self.current_seg = self.proba.argmax(0).cpu().numpy().transpose(2,1,0)

    def get_sliding_window(self,input,x,y,z,win_len = 192 ,win_h = 192 ,win_d = 48):
        """Return the sliding window for refinement. 
        patch size must be even number 
        input dim -> (c,d,h,w)"""

        self.pad_map = None
        self.i_pos = np.array([max(x - win_len//2,0) ,max(y - win_h//2,0),max(z - win_d//2,0)]) #lower bounds of the window
        self.f_pos = np.array([min(x + win_len//2,input.shape[3]),min(y + win_h//2,input.shape[2]),min(z + win_d//2,input.shape[1])]) #higher bounds

        sw = input[:, self.i_pos[2]: self.f_pos[2],
                    self.i_pos[1]: self.f_pos[1],
                    self.i_pos[0]: self.f_pos[0]] 

        if sw.shape != (win_len,win_h,win_d): # if true we need padding 
            left_exceed = np.array([x - win_len//2, y - win_h//2, z - win_d//2])
            right_exceed = np.array([x + win_len//2, y + win_h//2, z + win_d//2])
            pad_left = np.where(left_exceed < 0,-left_exceed,0)
            input_shape_reversed = (input.shape[3],input.shape[2],input.shape[1]) # needed to match with padding & exceed
            pad_right = np.where(right_exceed > input_shape_reversed,right_exceed -input_shape_reversed,0)
            self.pad_map = (pad_left[0],pad_right[0], # W
                            pad_left[1],pad_right[1], # H 
                            pad_left[2],pad_right[2], # D
                            0,0) #we don't pad channel dim ofc 
            padded_sw = pad(sw,self.pad_map,mode = 'constant',value = 0)

            assert padded_sw.shape == (input.shape[0],win_d,win_h,win_len), f"paddded sliding window does not have the correct shape.\n ->{padded_sw.shape} insted of {(input.shape[0],win_d,win_h,win_len)} "
            
            return padded_sw
        else :

            return sw 
        
    def refinement_prediction(self,input,x,y,z):
        """ predicting only a window around where you put the click"""
        window = self.get_sliding_window(input,x,y,z)
        with torch.no_grad():
            window_logits = self.predictor.network(window[None]).squeeze(0)
            self.output = window_logits.argmax(0).cpu().numpy()
            # self.output = torch.zeros(window_logits[0].shape,device = torch.device('cuda:0'))
            # window_proba= torch.softmax(window_logits,dim = 0 )
            # gaussian_weight = compute_gaussian(window_logits.shape,value_scaling_factor = 10) #weighting for the logits 
            # window_logits*=gaussian_weight

        if self.pad_map is not None :
            # unpad the image 

            self.output = self.output[self.pad_map[4]:self.output.shape[0] - self.pad_map[5],
                                      self.pad_map[2]:self.output.shape[1] - self.pad_map[3],
                                      self.pad_map[0]:self.output.shape[2] - self.pad_map[1]]
            
            # zero_map = torch.ones_like(window_logits,dtype=torch.bool)
            # zero_map[self.pad_map[4]:self.output.shape[0] - self.pad_map[5],
            #             self.pad_map[2]:self.output.shape[1] - self.pad_map[3],
            #             self.pad_map[0]:self.output.shape[2] - self.pad_map[1]] = False
            
            window_logits = window_logits[:,self.pad_map[4]:self.output.shape[0] - self.pad_map[5],
                                            self.pad_map[2]:self.output.shape[1] - self.pad_map[3],
                                            self.pad_map[0]:self.output.shape[2] - self.pad_map[1]]
            
        # Replace the old with prediction with the new one ? or should we do majority vote? # what about next predictions then ?????    
        
        self.current_seg[ self.i_pos[0]: self.f_pos[0],
                                     self.i_pos[1]: self.f_pos[1],
                                     self.i_pos[2]: self.f_pos[2]] = self.output.transpose(2,1,0)
        

        # window_proba= torch.softmax(window_logits,dim = 0 )
        # self.proba[:,self.pad_map[4]:self.output.shape[0] - self.pad_map[5],
        #                                     self.pad_map[2]:self.output.shape[1] - self.pad_map[3],
        #                                     self.pad_map[0]:self.output.shape[2] - self.pad_map[1]] += window_proba
                                                            
        # self.current_seg = self.proba.argmax(0).cpu().numpy().transpose(2,1,0)

    def get_sliding_window_2(self,input,x,y,z,win_len = 192 ,win_h = 192 ,win_d = 48):
        """Return the sliding window for refinement. 
        patch size must be even number 
        input dim -> (c,d,h,w)"""

        self.pad_map = None
        self.pad_legacy = None
        self.i_pos = np.array([max(x - win_len//2,0) ,max(y - win_h//2,0),max(z - win_d//2,0)]) #lower bounds of the window
        self.f_pos = np.array([min(x + win_len//2,input.shape[3]),min(y + win_h//2,input.shape[2]),min(z + win_d//2,input.shape[1])]) #higher bounds

        sw = input[:, self.i_pos[2]: self.f_pos[2],
                    self.i_pos[1]: self.f_pos[1],
                    self.i_pos[0]: self.f_pos[0]] 

        if sw.shape != (win_len,win_h,win_d): # if true we need padding 
            left_exceed = np.array([x - win_len//2, y - win_h//2, z - win_d//2])
            right_exceed = np.array([x + win_len//2, y + win_h//2, z + win_d//2])
            pad_left = np.where(left_exceed < 0,-left_exceed,0)
            input_shape_reversed = (input.shape[3],input.shape[2],input.shape[1]) # needed to match with padding & exceed
            pad_right = np.where(right_exceed > input_shape_reversed,right_exceed -input_shape_reversed,0)
            pad_right = np.where(pad_right > self.i_pos, self.i_pos,pad_right) # prevent of having negative bounds in self.extend_map
            self.pad_map = (pad_left[0],pad_right[0], # W
                            pad_left[1],pad_right[1], # H 
                            pad_left[2],pad_right[2], # D
                            0,0) #we don't pad channel dim ofc 
            self.extend_map = (self.i_pos[2] - pad_right[2] , self.f_pos[2] + pad_left[2],
                                self.i_pos[1] - pad_right[1] , self.f_pos[1] + pad_left[1],
                                self.i_pos[0] - pad_right[0] , self.f_pos[0] + pad_left[0]
                                )
                                    # instead of padding we extend on the other side 
            extended_sw = input[:,self.extend_map[0] : self.extend_map[1],
                                self.extend_map[2] : self.extend_map[3],
                                self.extend_map[4] : self.extend_map[5]]
            if extended_sw.shape != (input.shape[0],win_d,win_h,win_len):
                diff = np.array([win_d,win_h,win_len]) - extended_sw.shape[1:]
                pad_after = diff // 2 
                pad_before = diff // 2  + diff % 2
                padded_sw = torch.zeros((*input.shape[:-3],win_d,win_h,win_len),device = torch.device('cuda:0'))
                pad_after = padded_sw.shape[1:] - pad_after
                self.pad_legacy = np.array([pad_before,pad_after])
                padded_sw[:,pad_before[0]:pad_after[0],pad_before[1]:pad_after[1],pad_before[2]:pad_after[2]] = extended_sw

                assert padded_sw.shape == (input.shape[0],win_d,win_h,win_len), f" extended sliding window does not have the correct shape.\n ->{padded_sw.shape} instead of {(input.shape[0],win_d,win_h,win_len)} "
                return padded_sw

            assert extended_sw.shape == (input.shape[0],win_d,win_h,win_len), f" extended sliding window does not have the correct shape.\n ->{extended_sw.shape} instead of {(input.shape[0],win_d,win_h,win_len)} "
            
            return extended_sw
        else :

            return sw  
    
    def refinement_prediction_bis(self,input,x,y,z):
        """ predicting only a window around where you put the click"""
        window = self.get_sliding_window_2(input,x,y,z)
        with torch.no_grad():
            window_logits = self.predictor.network(window[None]).squeeze(0)
            self.output = window_logits.argmax(0).cpu().numpy()
            
        # Replace the old with prediction with the new one ? or should we do majority vote? # what about next predictions then ?????    
        if self.extend_map is not None : 

            self.current_seg[self.extend_map[4] : self.extend_map[5],
                                self.extend_map[2] : self.extend_map[3],
                                self.extend_map[0] : self.extend_map[1] ] = self.output.transpose(2,1,0)
        else:
            self.current_seg[ self.i_pos[0]: self.f_pos[0],
                                     self.i_pos[1]: self.f_pos[1],
                                     self.i_pos[2]: self.f_pos[2]] = self.output.transpose(2,1,0)
            
    
    def refinement_prediction_proba(self,input,x,y,z):
        """ predicting only a window around where you put the click & dealing with probablity"""
        window = self.get_sliding_window_2(input,x,y,z)
        with torch.no_grad():
            window_logits = self.predictor.network(window[None]).squeeze(0)
            window_proba = torch.softmax(window_logits,dim = 0 )
            gaussian_weight = make_gaussian(window_logits.shape[1:],(z + self.pad_map[4] - self.pad_map[5],y + self.pad_map[2] - self.pad_map[3],x + self.pad_map[0] - self.pad_map[1]),value_scaling_factor = 10, sigma_scale= 1./8) # weighting for the logits/proba
            gaussian_weight/=gaussian_weight.max()
            gaussian_weight[gaussian_weight == 0] = gaussian_weight[gaussian_weight != 0].min()
            # SCR.screen_gaussian_weight_map(gaussian_weight)
            window_proba *= gaussian_weight[None]
            self.output = window_logits.argmax(0).cpu().numpy()
        # Replace the old with prediction with the new one ? or should we do majority vote? # what about next predictions then ?????    
        if self.extend_map is not None : 
            self.proba[:,self.extend_map[0] : self.extend_map[1],
                                self.extend_map[2] : self.extend_map[3],
                                self.extend_map[4] : self.extend_map[5] ] = self.proba[:,self.extend_map[0] : self.extend_map[1],self.extend_map[2] : self.extend_map[3],self.extend_map[4] : self.extend_map[5]] * (1 - gaussian_weight) +  window_proba
        else:
            self.proba[:,self.i_pos[0]: self.f_pos[0],
                                     self.i_pos[1]: self.f_pos[1],
                                     self.i_pos[2]: self.f_pos[2]] = self.proba[:,self.i_pos[0]: self.f_pos[0],self.i_pos[1]: self.f_pos[1],self.i_pos[2]: self.f_pos[2]] * (1 - gaussian_weight) +  window_proba
        self.current_seg = self.proba.argmax(0).cpu().numpy().transpose(2,1,0)

    def refinement_prediction_logits(self,input,x,y,z):
        """ predicting only a window around where you put the click & dealing with probablity"""
        window = self.get_sliding_window_2(input,x,y,z)
        
        # SCR.screen_tensor(window[0],'sliding_window',point = (reshape_z,reshape_y,reshape_x))
        with torch.no_grad():
            window_logits = self.predictor.network(window[None]).squeeze(0)
            gaussian_weight = make_gaussian(window_logits.shape[1:],(z - self.extend_map[0],y - self.extend_map[2],x - self.extend_map[4]),value_scaling_factor = 10, sigma_scale= 1./8) # weighting for the logits/proba
            gaussian_weight/=gaussian_weight.max()
            gaussian_weight[gaussian_weight == 0] = gaussian_weight[gaussian_weight != 0].min()
            window_logits *= gaussian_weight[None]
            self.output = window_logits.argmax(0).cpu().numpy()
        # Replace the old with prediction with the new one ? or should we do majority vote? # what about next predictions then ?????    
        if self.extend_map is not None : 
            self.logits[:,self.extend_map[0] : self.extend_map[1],
                                self.extend_map[2] : self.extend_map[3],
                                self.extend_map[4] : self.extend_map[5] ] = self.logits[:,self.extend_map[0] : self.extend_map[1],self.extend_map[2] : self.extend_map[3],self.extend_map[4] : self.extend_map[5]] * (1 - gaussian_weight) +  window_logits
        else:
            self.logits[:,self.i_pos[0]: self.f_pos[0],
                                     self.i_pos[1]: self.f_pos[1],
                                     self.i_pos[2]: self.f_pos[2]] = self.logits[:,self.i_pos[0]: self.f_pos[0],self.i_pos[1]: self.f_pos[1],self.i_pos[2]: self.f_pos[2]] * (1 - gaussian_weight) +  window_logits
        self.current_seg = self.logits.argmax(0).cpu().numpy().transpose(2,1,0)
    

    def refinement_prediction_logits_manyfolds(self,input,x,y,z):
        """ predicting only a window around where you put the click & dealing with probablity"""

        window = self.get_sliding_window_2(input,x,y,z)
        aggregated_logits = torch.zeros(window.shape,device = torch.device('cuda:0'))[1:]
        #initialing network....
        for params in self.predictor.list_of_parameters:

            # messing with state dict names...
            if not isinstance(self.predictor.network, OptimizedModule):
                self.predictor.network.load_state_dict(params)
            else:
                self.predictor.network._orig_mod.load_state_dict(params)

            with torch.no_grad():
                window_logits = self.predictor.network(window[None]).squeeze(0)
                aggregated_logits += window_logits

        if len(self.predictor.list_of_parameters) > 1:
            aggregated_logits /= len(self.predictor.list_of_parameters)

        if self.pad_legacy is not None : # need to unpad the logits 
            window_logits = window_logits[:,self.pad_legacy[0][0] : self.pad_legacy[1,0],
                                          self.pad_legacy[0][1] : self.pad_legacy[1,1],
                                          self.pad_legacy[0][2] : self.pad_legacy[1,2]]
            
        gaussian_weight = make_gaussian(window_logits.shape[1:],(z - self.extend_map[0],y - self.extend_map[2],x - self.extend_map[4]),value_scaling_factor = 10, sigma_scale= 1./8) # weighting for the logits/proba
        gaussian_weight/=gaussian_weight.max()
        gaussian_weight[gaussian_weight == 0] = gaussian_weight[gaussian_weight != 0].min()
        window_logits *= gaussian_weight[None]
        self.output = window_logits.argmax(0).cpu().numpy()
        
        if self.extend_map is not None : 
            # SCR.compare_2_tensor(window_logits.cpu().numpy(),gaussian_weight[None].cpu().numpy(),'gaussien_weighted')
            self.logits[:,self.extend_map[0] : self.extend_map[1],
                                self.extend_map[2] : self.extend_map[3],
                                self.extend_map[4] : self.extend_map[5] ] = self.logits[:,self.extend_map[0] : self.extend_map[1],self.extend_map[2] : self.extend_map[3],self.extend_map[4] : self.extend_map[5]] * (1 - gaussian_weight) +  window_logits
        else:
            self.logits[:,self.i_pos[0]: self.f_pos[0],
                                    self.i_pos[1]: self.f_pos[1],
                                    self.i_pos[2]: self.f_pos[2]] = self.logits[:,self.i_pos[0]: self.f_pos[0],self.i_pos[1]: self.f_pos[1],self.i_pos[2]: self.f_pos[2]] * (1 - gaussian_weight) +  window_logits
                
        self.current_seg = self.logits.argmax(0).cpu().numpy().transpose(2,1,0)

        