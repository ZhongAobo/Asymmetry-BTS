import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math
import os

datapath = "/home/SHARE2/ZXY/RFNet/BRATS2020_Training_none_npy"

padding = 30        
move_range = 5    
rotate_range = 5       

def dice_loss(output, target, eps=1e-7):
    output = output.float()
    target = target.float()
    loss = 0.0
    for c in range(output.shape[2]):
        num = torch.sum(output[:,:,c] * target[:,:,c])
        l = torch.sum(output[:,:,c])
        r = torch.sum(target[:,:,c])
        dice = 2.0 * num / (l+r+eps)
        loss += 1.0 - 1.0 * dice
    return loss

def similarity_match_for_img(img,direction='vertical',min_threshold=100,max_treshold=150,kernel_size=2):

    H_original,W_original,C_original = img.shape
    # print(img.max(),img.min())
    if img.max()-img.min() != 0.0:
        input = ((img-img.min())/(img.max()-img.min())*255).astype(np.uint8)    
    else:
        input = ((img-img.min())/255).astype(np.uint8)     



    kernel = np.ones((kernel_size,kernel_size), np.uint8)

    for i in range(C_original):
        input[:,:,i] = cv2.Canny(input[:,:,i],min_threshold,max_treshold)
        input[:,:,i] = cv2.dilate(input[:,:,i],kernel)              

    input = torch.from_numpy(input)                            
    input[input!=0] = 1

    best_loss_1 = best_loss_2 = 1e4                                       
    vertical = horizontal = rotate = 0                                
    
    best_shift_1 = best_shift_2 = None                               
    best_rotate = None

    if H_original>W_original:
        diff = H_original-W_original                  
        square_input = torch.zeros([H_original,H_original,C_original])
        square_input[:,diff//2:W_original+diff//2,:] = input
        input = square_input
    if H_original<W_original:
        diff = W_original-H_original
        square_input = torch.zeros([W_original,W_original,C_original])
        square_input[diff//2:H_original+diff//2,:,:] = input
        input = square_input 

    H,W,C = input.shape     
    if direction == 'vertical':
        dims = 0
        for i in range(-(move_range),move_range):
            expanded_input = torch.zeros((H+abs(i),W,C))
            if i>=0:
                expanded_input[i:,:,:] = input    
                shifted_input = expanded_input[:H,:,:]
            else:
                expanded_input[:H,:,:] = input    
                shifted_input = expanded_input[i:,:,:]
            shifted_flip = torch.flip(shifted_input,[dims])
            loss = dice_loss(shifted_input,shifted_flip)/C
            if loss < best_loss_1:
                best_loss_1 = loss
                vertical = i
                best_shift_1 = shifted_input

        temp = best_shift_1.permute(2,0,1)
        temp = temp.unsqueeze(0)
        best_loss_2 = best_loss_1
        for theta in range(-rotate_range,rotate_range+1):
            angle = theta/180.0*math.pi
            transform_matrix = torch.tensor([[math.cos(angle),math.sin(-angle),0],[math.sin(angle),math.cos(angle),0]])
            transform_matrix = transform_matrix.unsqueeze(0)

            grid = F.affine_grid(transform_matrix,temp.shape,align_corners=False)

            first_layer = F.grid_sample(temp[:,0:1,:,:],grid,mode='bicubic',align_corners=False)
            rotated_input = first_layer.repeat(1,C,1,1)
            for c in range(1,C):
                rotated_input[:,c:c+1,:,:] = F.grid_sample(temp[:,c:c+1,:,:],grid,mode='bicubic',align_corners=False)
            rotated_input = rotated_input.squeeze(0)
            rotated_input = rotated_input.permute(1,2,0)
            rotated_flip = torch.flip(rotated_input,[dims])
            loss = dice_loss(shifted_input,shifted_flip)/C
            if loss < best_loss_2:
                best_loss_2 = loss
                rotate = theta
                best_rotate = rotated_input


    elif direction == 'horizontal':
        dims = 1
        for i in range(-(move_range),move_range):
            expanded_input = torch.zeros((H,W+abs(i),C))
            if i>=0:
                expanded_input[:,i:,:] = input      #右移
                shifted_input = expanded_input[:,:W,:]
            else:
                expanded_input[:,:W,:] = input      #左移
                shifted_input = expanded_input[:,i:,:]
            shifted_flip = torch.flip(shifted_input,[dims])
            loss = dice_loss(shifted_input,shifted_flip)/C
            if loss < best_loss_1:
                best_loss_1 = loss
                vertical = i
                best_shift_1 = shifted_input

        temp = best_shift_1.permute(2,0,1)
        temp = temp.unsqueeze(0)
        best_loss_2 = best_loss_1
        for theta in range(-rotate_range,rotate_range+1):
            angle = theta/180.0*math.pi
            transform_matrix = torch.tensor([[math.cos(angle),math.sin(-angle),0],[math.sin(angle),math.cos(angle),0]])
            transform_matrix = transform_matrix.unsqueeze(0)

            grid = F.affine_grid(transform_matrix,temp.shape,align_corners=False)

            first_layer = F.grid_sample(temp[:,0:1,:,:],grid,mode='bicubic',align_corners=False)
            rotated_input = first_layer.repeat(1,C,1,1)
            for c in range(1,C):
                rotated_input[:,c:c+1,:,:] = F.grid_sample(temp[:,c:c+1,:,:],grid,mode='bicubic',align_corners=False)
            rotated_input = rotated_input.squeeze(0)
            rotated_input = rotated_input.permute(1,2,0)
            rotated_flip = torch.flip(rotated_input,[dims])
            loss = dice_loss(shifted_input,shifted_flip)/C
            if loss < best_loss_2:
                best_loss_2 = loss
                rotate = theta
                best_rotate = rotated_input


    return vertical,horizontal,rotate                            

def position_recover_for_img(img,vertical,horizontal,rotate):
    input = torch.from_numpy(img)
    H_original,W_original,C = input.shape
    diff = 0

    if H_original>W_original:
        expanded_input = torch.zeros([H_original,H_original,C])
        diff = H_original-W_original
        for c in range(C):
            expanded_input[:,:,c] = F.pad(input[:,:,c],(diff//2,diff-diff//2,0,0),'constant',img.min())
        input = expanded_input

    if H_original<W_original:
        expanded_input = torch.zeros([W_original,W_original,C])
        diff = W_original-H_original
        for c in range(C):
            expanded_input[:,:,c] = F.pad(input[:,:,c],(0,0,diff//2,diff-diff//2),'constant',img.min())
        input = expanded_input

    H,W,C = input.shape
    input = input.permute(2,0,1)    #[C,H,W]
    input = input.unsqueeze(0)      #[1,C,H,W]
    angle = rotate/180*math.pi
    transform_matrix = torch.tensor([[math.cos(angle),math.sin(-angle),0],[math.sin(angle),math.cos(angle),0]])
    transform_matrix = transform_matrix.unsqueeze(0)
    grid = F.affine_grid(transform_matrix,input.shape,align_corners=False)

    first_layer = F.grid_sample(input[:,0:1,:,:],grid,mode='bicubic',align_corners=False)
    rotated_input = first_layer.repeat(1,C,1,1)
    for c in range(1,C):
        rotated_input[:,c:c+1,:,:] = F.grid_sample(input[:,c:c+1,:,:],grid,mode='bicubic',align_corners=False)

    rotated_input = rotated_input.squeeze(0)    #[C,H,W]
    input = rotated_input.permute(1,2,0)       #[H,W,C]

    if horizontal > 0:
        expanded_input = torch.zeros((H,W+horizontal,C))
        expanded_input += input.min()
        expanded_input[:,horizontal:,:] = input
        input = expanded_input[:,:W,:]
    elif horizontal < 0:
        expanded_input = torch.zeros((H,W-horizontal,C))
        expanded_input += input.min()
        expanded_input[:,:W,:] = input
        input = expanded_input[:,(-horizontal):,:]

    if vertical > 0:
        expanded_input = torch.zeros((H+vertical,W,C))
        expanded_input += input.min()
        expanded_input[vertical:,:,:] = input
        input = expanded_input[:H,:,:]
    elif vertical < 0:
        expanded_input = torch.zeros((H-vertical,W,C))
        expanded_input += input.min()
        expanded_input[:H,:,:] = input
        input = expanded_input[(-vertical):,:,:]
    
    if H_original>=W_original:
        output = input[:,diff//2:H-(diff-diff//2),:]
    elif H_original<W_original:
        output = input[diff//2:H-(diff-diff//2),:,:]

    output = output.detach().numpy()

    return output

def position_adjust_for_img(img,vertical,horizontal,rotate):
    input = torch.from_numpy(img)   #H,W,C
    H_original,W_original,C = input.shape
    diff = 0
    if H_original>W_original:
        expanded_input = torch.zeros([H_original,H_original,C])
        diff = H_original-W_original
        for i in range(0,C):
            expanded_input[:,:,i] = F.pad(input[:,:,i],(diff//2,diff-diff//2,0,0),'constant',img.min())
        input = expanded_input

    if H_original<W_original:
        expanded_input = torch.zeros([W_original,W_original,C])
        diff = W_original-H_original
        for i in range(0,C):
            expanded_input[:,:,i] = F.pad(input[:,:,i],(0,0,diff//2,diff-diff//2),'constant',img.min())
        input = expanded_input

    H,W,C = input.shape

    if vertical > 0:
        expanded_input = torch.zeros((H+abs(vertical),W,C))
        expanded_input += input.min()
        expanded_input[vertical:,:,:] = input
        input = expanded_input[:H,:,:]
    elif vertical < 0:
        expanded_input = torch.zeros((H+abs(vertical),W,C)) 
        expanded_input += input.min()
        expanded_input[:H,:,:] = input
        input = expanded_input[abs(vertical):,:,:]
    
    if horizontal > 0:
        expanded_input = torch.zeros((H,W+abs(horizontal),C)) 
        expanded_input += input.min()
        expanded_input[:,horizontal:,:] = input
        input = expanded_input[:,:W,:]
    elif horizontal < 0:
        expanded_input = torch.zeros((H,W+abs(horizontal),C))
        expanded_input += input.min()
        expanded_input[:,:W,:] = input
        input = expanded_input[:,abs(horizontal):,:]

    input = input.permute(2,0,1)
    input = input.unsqueeze(0)#[1,C,H,W]
    angle = rotate/180*math.pi
    transform_matrix = torch.tensor([[math.cos(angle),math.sin(-angle),0],[math.sin(angle),math.cos(angle),0]])
    transform_matrix = transform_matrix.unsqueeze(0)
    grid = F.affine_grid(transform_matrix,input.shape,align_corners=False)

    first_layer = F.grid_sample(input[:,0:1,:,:],grid,mode='bicubic',align_corners=False)
    rotated_input = first_layer.repeat(1,C,1,1)
    for c in range(1,C):
        rotated_input[:,c:c+1,:,:] = F.grid_sample(input[:,c:c+1,:,:],grid,mode='bicubic',align_corners=False)

    rotated_input = rotated_input.squeeze(0)
    output = rotated_input.permute(1,2,0)

    if H_original>=W_original:
        output = output[:,diff//2:H-(diff-diff//2),:]
    elif H_original<W_original:
        output = output[diff//2:H-(diff-diff//2),:,:]

    output = output.detach().numpy()

    return output

def create_diff_diagram(img,direction='vertical'):

    if direction == 'vertical':
        dims = 0
    elif direction == 'horizontal':
        dims = 1
    else:
        print('direction is error')
        exit(0)

    H,W,C = img.shape             
    vertical,horizontal,rotate = similarity_match_for_img(img,direction)

    pad = np.zeros([2*padding+H,2*padding+W,C]).astype(np.float32)
    pad.fill(img.min())                                              
    pad[padding:padding+H,padding:padding+W,:] = img             

    output = position_adjust_for_img(pad,vertical,horizontal,rotate) 

    output_flip = torch.flip(torch.from_numpy(output),[dims])
    diff = torch.from_numpy(output) - output_flip                  
        
    diff = diff.detach().numpy()
    diff = position_recover_for_img(diff,-vertical,-horizontal,-rotate) 

    diff = diff[padding:padding+H,padding:padding+W,:]                                                

    return diff                                                  

def main(train_file: str):
    data_file_path = os.path.join(datapath, train_file)
    with open(data_file_path, 'r') as f:
        datalist = [i.strip() for i in f.readlines()]
    datalist.sort()

    for dataname in datalist:
        x_path = os.path.join(datapath, 'vol', dataname+'_vol.npy')
        diff_path = os.path.join(datapath, 'diff', dataname+'_diff.npy')
        x = np.load(x_path)

        diff = np.zeros_like(x).astype(np.float32)

        for j in range(0,x.shape[3]):
            for i in range(0,x.shape[2]):
                diff_map = x[:,:,i:i+1,j]
                try:
                    diff[:,:,i:i+1,j] = create_diff_diagram(diff_map,'vertical')
                except:
                    continue

        np.save(diff_path,diff)
        print(diff_path,"has been saved...")


if __name__ == '__main__': 
    diff_path = os.path.join(datapath, "diff")
    os.makedirs(diff_path, exist_ok=True)
    main("train.txt")
    main("test.txt")
    main("val.txt")