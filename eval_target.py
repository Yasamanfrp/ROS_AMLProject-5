import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import random
import tensorflow as tf
import os


#### Implement the evaluation on the target for the known/unknown separation

def evaluation(args,feature_extractor,rot_cls,target_loader_eval,device):

    feature_extractor.eval()
    rot_cls.eval()
  #'''calculate the normality score at the end of Stage I of ROS!
   #Normality Score Pseudo-code'''

    tot_img = len(target_loader_eval.dataset) #Total images
    entropy = torch.zeros((tot_img,4), device=device)
    RotationScore = torch.zeros((tot_img,4), device=device)

    ground_truth = torch.zeros((tot_img,), device=device)
    unknown_pred = torch.zeros((tot_img,), device=device)
    normality_score = torch.zeros((tot_img,), device=device)
    

    log_classes_known = (torch.as_tensor(np.log(args.n_classes_known),dtype=torch.float32)).to(device)
    with torch.no_grad():
        for it, (data,class_l,data_rot,rot_l) in enumerate(target_loader_eval):
            data, class_l, data_rot, rot_l = data.to(device), class_l.to(device), data_rot.to(device), rot_l.to(device)
            
            features_original = feature_extractor(data)
    #'''nomenclature of the variables are based on the literature
    #Bucci, S., Loghmani, M. R., Tommasi, T.: On the effectiveness of
    #image rotation for open set domain adaptation. In: ECCV (2020)''' 
            #initialization
            zi_tot= torch.zeros((1,4), device=device) #entropy loss(score) as a function of zi==>(H(zi))
            zi_tot=torch.as_tensor(zi_tot, dtype=torch.float32, device=device)
            H_sigma = torch.zeros((1,4), device=device) #relative rotation variants of a target sample
            H_sigma = torch.as_tensor(H_sigma, dtype=torch.float32, device=device)
            for r in range(4):
                data_rot = torch.rot90(data, k=r, dims=[2, 3]) #compute roation
                rot_l = r
                features_rotated = feature_extractor(data_rot)
                rot_pred = rot_cls(torch.cat((features_original,features_rotated),1))
                zi = torch.nn.Softmax(dim=-1)(rot_pred)
                h=(zi)*torch.log((zi) + 1e-21)/log_classes_known
                # z_i = torch.nn.Softmax(dim=0)(torch.flatten (
                #     rot_cls(torch.cat([features_original,features_rotated],1))))

                # h=[r] = (z_i.dot(torch.log10(z_i))/log(args.n_classes_known, 10)).item()
                # rotation_scores[r] = z_i[r] 
                zi_tot+=zi
                H_sigma +=h

            RotationScore[it] = zi_tot
            entropy[it] = (H_sigma/4)    #4=|r|

            if (class_l >= args.n_classes_known):
                ground_truth[it] = 0
            else:
                ground_truth[it] = 1

"normality score is a function of the ability of the network to correctly"
"predict the semantic class and orientation of a target sample known as Rotation Score"
"and its confidence evaluated on the basis of the prediction entropy known as Entropy Score"
            
    #normality score= Max(Rotation score, 1 - mean(entropy losse)) 
    RS_std,RS_mean = torch.std_mean(RotationScore)
    RotationScore = (RotationScore - RS_mean)/RS_std
    EntropyScore = 1 - entropy
    ES_std,ES_mean = torch.std_mean(EntropyScore)
    EntropyScore = (EntropyScore - ES_mean)/ES_std
    normality_score = torch.max(RotationScore,EntropyScore)
    normality_score = torch.mean(normality_score,dim=1)
    NS_std,NS_mean = torch.std_mean(normality_score)
    normality_score = (normality_score - NS_mean)/NS_std
    

    for it1 in range(len(normality_score)):
        if (normality_score[it1] > args.threshold):
            unknown_pred[it1] = 1
        else:
            unknown_pred[it1] = 0

    normality_score_cpu = normality_score.cpu()
    ground_truth_cpu = ground_truth.cpu()
    normality_score_cpu = np.array(normality_score_cpu)
    ground_truth_cpu = np.array(ground_truth_cpu)

    auroc = roc_auc_score(ground_truth_cpu,normality_score_cpu)
    print('   AUROC %.4f' % auroc)

    
    "Update the known and unknown samples at the source and target files" 
    
    number_of_known_samples = (torch.sum(unknown_pred)).int()
    number_of_unknown_samples = (tot_img - torch.sum(unknown_pred)).int()

    print('   The number of target samples selected as known is: ', number_of_known_samples,' of ',torch.sum((ground_truth),0).int())
    print('   The number of target samples selected as unknown is: ', number_of_unknown_samples, ' of ', (tot_img - (torch.sum((ground_truth),0))).int())

    images_labels = target_loader_eval.dataset.labels 
    images_paths = target_loader_eval.dataset.names

    known_images_paths, known_images_labels, unknown_images_paths = [], [], []

    # Known/unknown paths and labels for evaluated images
    for it2 in range(tot_img):
        if (unknown_pred[it2]==1):
            known_images_labels = known_images_labels+[images_labels[it2]]
            known_images_paths = known_images_paths+[images_paths[it2]]
        else:
            unknown_images_paths = unknown_images_paths+[images_paths[it2]]

    # Create new txt files
    rand = random.randint(0,100000)
    rand = 25
    print('   Generated random number is :', rand)
    if not os.path.isdir('/content/drive/MyDrive/ROS_mich/mich/new_txt_list/'):
        os.mkdir('/content/drive/MyDrive/ROS_mich/mich/new_txt_list/')

    # This txt files will have the names of the source images and the names of the target images selected as unknown
    # According to bibliography: [Ds_known + Dt_unknown], Currently implemented: [Ds_known + Dt_unknown]
    mypath= "/content/drive/MyDrive/ROS_mich/mich/"
    target_unknown = open(mypath + 'new_txt_list/'+args.source+'_unknown_'+str(rand)+'.txt','w')
    for it3 in range (number_of_unknown_samples):
        target_unknown.write(unknown_images_paths[it3]+' '+str(args.n_classes_known)+'\n')
        # Label will always be 45, since that indicates the unknown class label

    files = [mypath + 'txt_list/'+args.source+'_known'+'.txt',mypath + 'new_txt_list/'+args.source+'_unknown_'+str(rand)+'.txt']
    # Open file3 in write mode
    with open(mypath + 'new_txt_list/'+args.source+'_known_'+str(rand)+'.txt', 'w') as outfile:
        for names in files:
            # Open each file in read mode
            with open(names) as infile:
                # read the data from Ds and Dt_unknown and write [Ds U Dt_unknown]
                outfile.write(infile.read())

    # This txt files will have the names of the target images selected as known
    # According to bibliography:[Dt_known], Currently implemented:[Dt_known]
    target_known = open(mypath + 'new_txt_list/'+args.target+'_known_'+str(rand)+'.txt','w')
    for it4 in range (number_of_known_samples):
        target_known.write(known_images_paths[it4]+' '+str(known_images_labels[it4])+'\n')

    return rand



