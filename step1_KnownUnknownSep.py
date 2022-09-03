
import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler
from center_loss import CenterLoss
#### Implement Step1
#we need to run the Epoch and get the losses and accuracies
def _do_epoch(args,feature_extractor,rot_cls,obj_cls,cent_loss,source_loader,optimizer,device):
    "feature_extractor= features_original that get extracted from the images "
    " rotation and object classifiers is needed"
    "Dataloader of the Source domain is needed"
    "Gradient optimizer is needed"

    #in order to prepare the trainting and classification we need to compute the loss function
    #set up
    criterion = nn.CrossEntropyLoss() # for classification, we use Cross Entropy
    feature_extractor.train()
    obj_cls.train()
    rot_cls.train()

    # Useful variables and initializations
    k = len(source_loader.dataset) # Number of images
    #initializations in order to start iteration over dataset
    cls_corrects, rot_corrects, iterations  = 0, 0, 0

    
    for it, (data, class_l, data_rot, rot_l) in enumerate(source_loader):
        data, class_l, data_rot, rot_l  = data.to(device), class_l.to(device), data_rot.to(device), rot_l.to(device)

        #manually set the gradients to Zero before starting a new iteration
        optimizer.zero_grad()

        # Extract features_original(in training)
        features_original = feature_extractor(data)
        features_original_rot = feature_extractor(data_rot)
        
        #pas features to Semantic classifier
        sem_cls = obj_cls(features_original)

        #'training uses the concatenated features of the original and rotated image'
        
        features_original_rot = torch.cat((features_original, features_original_rot),1)
        #'the network will learn the relative orientation of the objects'
        #rotation recognition
        recognition_rot = rot_cls(features_original_rot)

        #'we train two loss function'
        #first:loss function for object recognition
        class_loss  = criterion(sem_cls, class_l)

        #second:loss function for rotation recognition
        rot_loss    = criterion(recognition_rot, rot_l)
        
        cnt_loss    = (cent_loss(features_original, class_l))
        

        loss = class_loss + args.weight_RotTask_step1*rot_loss + args.weight_Center_Loss*cnt_loss
        loss.backward() #backward pass: compute gradients

        # remove the effect of the hyperparameter on the adjustment of the center
        #the same learning rate is used for both networks and center loss 
        if (args.weight_Center_Loss > 0):
            for param in cent_loss.parameters():
                param.grad.data *= (1/(args.weight_Center_Loss))

        optimizer.step()
        
       #Find the highest "probability" corresponds to its index
       #soft max or Argmax?
        cls_pred = nn.Softmax(dim=-1)(sem_cls)
        rot_pred = nn.Softmax(dim=-1)(recognition_rot)

        #Get predictions
        _, cls_pred = torch.max(cls_pred, dim=1)
        _, rot_pred = torch.max(rot_pred, dim=1)

        # Update corrects
        running_corrects_cls += torch.sum(cls_pred == class_l)
        running_corrects_rot += torch.sum(rot_pred == rot_l)

        iterations += 1

    
    
    
    
    
    print('End of the images for cycle with %d iterations' % iterations)
    print('   Correct class predictions %d' % cls_corrects,'of',float(k))
    print('   Correct rot predictions %d' % rot_corrects,'of',float(k))

    #Apply accuracy equation
    acc_cls = running_corrects_cls/float(k)
    acc_cls = acc_cls*100
    acc_rot = running_corrects_rot/float(k)
    acc_rot = acc_rot*100

    return class_loss, acc_cls, rot_loss, acc_rot


def step1(args,feature_extractor,rot_cls,obj_cls,source_loader,device):
    cent_loss = CenterLoss(num_classes = args.n_classes_known, feat_dim = 512, use_gpu=True)
    optimizer, scheduler = get_optim_and_scheduler(feature_extractor, rot_cls, obj_cls, cent_loss, args.weight_Center_Loss, args.epochs_step1, args.learning_rate, args.train_all)

    for epoch in range(args.epochs_step1):
        class_loss, acc_cls, rot_loss, acc_rot, cnt_loss = _do_epoch(args,feature_extractor,rot_cls,obj_cls,cent_loss,source_loader,optimizer,device)
        print("   Class Loss %.4f, Class Accuracy %.4f,Rot Loss %.4f, Rot Accuracy %.4f, Center-Loss %.4f" % (class_loss.item(), acc_cls, rot_loss.item(), acc_rot, cnt_loss.item()))
        scheduler.step()