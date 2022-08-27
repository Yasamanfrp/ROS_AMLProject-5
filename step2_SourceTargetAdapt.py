
import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler
from itertools import cycle
import numpy as np


#### Implement Step2

def _do_epoch(args,feature_extractor,rot_cls,obj_cls,source_loader,target_loader_train,target_loader_eval,optimizer,device):

    criterion = nn.CrossEntropyLoss()
    feature_extractor.train()
    obj_cls.train()
    rot_cls.train()
    
    j = len(source_loader.dataset)       # Number of images of source dataset
    j = len(target_loader_train.dataset) # Number of images of train target dataset
     

    #The cycle() function accepts an iterable and generates an iterator
    target_loader_train = cycle(target_loader_train)
    '''The unknown part of the target will be used to train the unknown class
    known part instead will be used for the source-target adaptation
    the adaptation step can be performed only between the source domain
    and the known part of the target domain'''

    for it, (data_source, class_l_source, _, _) in enumerate(source_loader):

        # run over source and target at the same time
        (data_target, _, data_target_rot, rot_l_target) = next(target_loader_train)

        data_source, class_l_source  = data_source.to(device), class_l_source.to(device)
        data_target, data_target_rot, rot_l_target  = data_target.to(device), data_target_rot.to(device), rot_l_target.to(device)

        optimizer.zero_grad()

        source_features = feature_extractor(data_source)
        target_features = feature_extractor(data_target)
        target_features_rot = feature_extractor(data_target_rot)
        target_features_rot = torch.cat((target_features,target_features_rot),1)

        # Train the classifier
        '''the unknown
        dataset created from the target domain is combined
        with the samples from the source domain as an additional class.
        The object classifier will trained to recognize samples
        belonging to the unknown class
        '''
        sem_un_cls = obj_cls(source_features)
        recognition_un_rot = rot_cls(target_features_rot)

        class_loss = criterion(sem_un_cls, class_l_source)
        rot_loss = criterion(recognition_un_rot, rot_l_target)

        loss = class_loss + args.weight_RotTask_step2*rot_loss

        loss.backward()

        optimizer.step()

        cls_pred = nn.Softmax(dim=-1)(sem_un_cls)
        rot_pred = nn.Softmax(dim=-1)(recognition_un_rot)

        _, cls_pred = torch.max(sem_un_cls, dim=1)
        _, rot_pred = torch.max(recognition_un_rot, dim=1)

        cls_corrects += torch.sum(cls_pred == class_l_source)
        rot_corrects += torch.sum(rot_pred == rot_l_target)

    # Accuracy calculation
    acc_cls = float(cls_corrects)/float(j)

    acc_rot = float(rot_corrects)/float(j)
    
    print("   Class Loss %.4f, Class Accuracy %.4f,Rot Loss %.4f, Rot Accuracy %.4f" % (class_loss.item(), acc_cls, rot_loss.item(), acc_rot))

    #### Implement the final evaluation step, computing OS*, UNK and HOS
    feature_extractor.eval()
    obj_cls.eval()
    rot_cls.eval()

    # Correct prediction counters
    correct_classes_known = 0
    correct_classes_unknown = 0
    total_classes_known = 0
    total_classes_unknown = 0

    with torch.no_grad():
        for it, (data, class_l, _, _) in enumerate(target_loader_eval):
            data, class_l = data.to(device), class_l.to(device)
            features = feature_extractor(data)
            sem_cls = obj_cls(features) #Pass features to classifiers
            # Get predictions
            cls_pred = nn.Softmax(dim=1)(sem_cls)

            # Update counters
            if class_l >= args.n_classes_known:
                total_classes_unknown += 1
                correct_classes_unknown += torch.sum(
                    cls_pred == args.n_classes_known
                ).item()
            else:
                total_classes_known += 1
                correct_classes_known += torch.sum(
                    cls_pred == class_l
                ).item()

        #accuracies
        #The accuracy of the object classifier in recognizing the known category
        acc_known = correct_classes_known / total_classes_known  # OS*

        #The accuracy of the object classifier in recognizing the unknown category
        acc_unknown = correct_classes_unknown / total_classes_unknown  # UNK

        #the harmonic mean between OS* and UNK
        hos = (
            2 * acc_known * acc_unknown / (acc_known + acc_unknown)
        ) 
        print(
            "\nEvaluation: OS* %.4f, UNK %.4f, HOS %.4f" % (acc_known, acc_unknown, hos)
        )
    


def step2(args,feature_extractor,rot_cls,obj_cls,source_loader,target_loader_train,target_loader_eval,device):
    cent_loss = CenterLoss(num_classes = args.n_classes_known, feat_dim = 512, use_gpu=True)
    optimizer, scheduler = get_optim_and_scheduler(feature_extractor,rot_cls,obj_cls, cent_loss, 0, args.epochs_step2, args.learning_rate, args.train_all)


    for epoch in range(args.epochs_step2):
        print("Epoch: ", epoch, "----------------------------------------")
        _do_epoch(args,feature_extractor,rot_cls,obj_cls,source_loader,target_loader_train,target_loader_eval,optimizer,device)
        '''if epoch % 10 == 0:
            if not os.path.isdir("weights"):
                os.mkdir("weights")
            torch.save(
                feature_extractor.state_dict(),
                f"weights/step2_feature_extractor_{epoch}.pth",
            )
            torch.save(
                obj_cls.state_dict(), f"weights/step2_object_classifier_{epoch}.pth"
            )
            torch.save(
                rot_cls.state_dict(), f"weights/step2_rotation_classifier_{epoch}.pth"
            )'''
        scheduler.step()
