
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import random


#### Implement the evaluation on the target for the known/unknown separation

def evaluation(args,feature_extractor,rot_cls,target_loader_eval,device):

    feature_extractor.eval()
    rot_cls.eval()
'''calculate the normality score at the end of Stage I of ROS!
   Normality Score Pseudo-code'''

    normality_score = torch.empty(
    size=[len(target_loader_eval.dataset)],dtype=torch.float32) # Number of images 

    ground_truth = torch.empty(
        size=[len(target_loader_eval.dataset)], dtype=torch.int32)

    with torch.no_grad():
        for it, (data,class_l,data_rot,rot_l) in enumerate(target_loader_eval):
            data, class_l, data_rot, rot_l = data.to(device), class_l.to(device), data_rot.to(device), rot_l.to(device)
            
            features_original = feature_extractor(data)
    '''nomenclature of the variables are based on the literature
    Bucci, S., Loghmani, M. R., Tommasi, T.: On the effectiveness of
    image rotation for open set domain adaptation. In: ECCV (2020)''' 
            #initialization
            h= torch.zeros([4]) #entropy loss(score) as a function of zi==>(H(zi))
            rotation_scores = torch.zeros([4]) #relative rotation variants of a target sample
            for r in range(4):
                data_rot = torch.rot90(data, k=r, dims=[2, 3]) #compute roation
                features_rotated = feature_extractor(data_rot)
                z_i = torch.nn.Softmax(dim=-1)(
                    rot_cls(torch.cat((features,features_rot),1)))

                h=[r] = (
                    z_i.dot(torch.log10(z_i))
                    / log(args.n_classes_known, 10)
                ).item()
                rotation_scores[r] = z_i[r] 
            
'''normality score is a function of the ability of the network to correctly
predict the semantic class and orientation of a target sample known as Rotation Score,
and its confidence evaluated on the basis of the prediction entropy known as Entropy Score'''
            
            #normality score= Max(Rotation score, 1 - mean(entropy losse)) 
            
            normality_score[it] = torch.max(rotation_scores,1-torch.mean(h))

            #indicating known/unknown
            ground_truth[it] = 0 if class_l >= args.n_classes_known else 1

    # compute the AUROC score from the vector of target labels and the vector of normality scores. AUROC MUST be >0.5
    auroc = roc_auc_score(ground_truth, normality_score)  # type: float
    print("AUROC %.4f" % auroc)

    # create new txt files
    rand = random.randint(0, 100000)
    print("Generated random number is :", rand)

    if not os.path.isdir("new_txt_list"):
        os.mkdir("new_txt_list")

    # This txt files will have the names of the source images and the names of the target images selected as unknown
    source_unknown_path = shutil.copyfile(
        f"txt_list/{args.source}_known.txt",
        f"new_txt_list/{args.source}_known_{rand}.txt",
    )
    target_unknown = open(source_unknown_path, "a+")
    target_unknown.write(f"\n")# new line at the end of source images

    # This txt files will have the names of the target images selected as known
    target_known = open(f"new_txt_list/{args.target}_known_{rand}.txt", "w+")

    number_of_known_samples = 0
    number_of_unknown_samples = 0
    with torch.no_grad():
        for img_i, (_, class_l, _, _) in enumerate(target_loader_eval):
            if normality_score[img_i] >= args.threshold:
                # known domain
                target_known.write(
                    f"{target_loader_eval.dataset.names[img_i]} {class_l.item()}\n"
                )
                number_of_known_samples += 1
            else:
                #UNknown domain
                target_unknown.write(
                    f"{target_loader_eval.dataset.names[img_i]} {args.n_classes_known}\n"
                )
                number_of_unknown_samples += 1

    target_known.close()
    target_unknown.close()

    print(
        "The number of target samples selected as known is: ", number_of_known_samples
    )
    print(
        "The number of target samples selected as unknown is: ",
        number_of_unknown_samples,
    )

    return rand

