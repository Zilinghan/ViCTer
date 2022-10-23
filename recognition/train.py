import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import TripletMarginLoss
from torch.optim.lr_scheduler import LambdaLR
from yolov5.utils.general import LOGGER
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1
from recognition.dataset import TrainingSetLabeled, TrainingSetUnlabeled
from recognition.embedding import EmbeddingPool



def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def evaluate_accuracy(net, test_iter, device):
    if isinstance(net, torch.nn.Module):
        net.eval()
    total = 0
    total_num = 0
    
    max_prob_total = 0
    max_prob_lower_bound = 1
    
    false_max_prob_total = 0
    false_total_num = 0
    false_max_prob_upper_bound = 0
    
    with torch.no_grad():
        for X, y in test_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            
            psuedo_label = torch.softmax(y_hat, axis=1)
            max_prob, y_hat = torch.max(psuedo_label, axis=1)
            
            # Check the prob
            max_prob_total += float(max_prob.sum())
            
            cmp = y_hat.type(y.dtype) == y
            total += float(cmp.type(y.dtype).sum())
            total_num += y.numel()
            
            # Check the false detection
            false_max_prob_total += float(max_prob[~cmp].sum())
            false_total_num += float((~cmp).sum())
            
            # Check the upper bound for false detection
            try:
                false_upper = max_prob[~cmp].max().item()
                if false_upper > false_max_prob_upper_bound:
                    false_max_prob_upper_bound = false_upper
            except:
                pass
                
            # Check the lower bound for correct detection
            try:
                correct_lower = max_prob[cmp].min().item()
                if correct_lower < max_prob_lower_bound:
                    max_prob_lower_bound = correct_lower
            except:
                pass
            
    try:
        return total/total_num, max_prob_total/total_num, false_max_prob_total/false_total_num, \
           false_max_prob_upper_bound, max_prob_lower_bound
    except:
        return total/total_num, max_prob_total/total_num, 0, \
           0, max_prob_lower_bound


def train_SSL(model, labeled_trainloader, unlabeled_trainloader, num_epochs, num_iters, learning_rate, temp=1, threshold=0.95):
    # params_1x: parameters with 1x learning rates
    params_1x = [param for name, param in model.named_parameters() if name not in ["logits.weight", "logits.bias"]]
    grouped_parameters = [
        {'params': params_1x, 'lr': learning_rate},
        {'params': model.logits.parameters(), 'lr': learning_rate*180}
    ]
    optimizer = torch.optim.SGD(grouped_parameters, lr=learning_rate, weight_decay=0.001)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_epochs*num_iters)
    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    
    device = get_device()
    
    model.train()
    for epoch in range(num_epochs):
        for batch_idx in range(num_iters):
            # obtain the labeled batch
            try:
                inputs_x, targets_x = next(labeled_iter)
            except:
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_iter)
            
            # obtain te unlabeled batch
            try:
                inputs_u, inputs_u_w = next(unlabeled_iter)
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                inputs_u, inputs_u_w = next(unlabeled_iter)
                
            num_x = inputs_x.shape[0] # number of labeled images in a batch
            num_u = inputs_u.shape[0] # number of unlabeled imaegs in a batch
            
            # obtain the outputs of the model
            inputs = torch.cat((inputs_x, inputs_u, inputs_u_w)).to(device)
            targets_x = targets_x.to(device)
            logits = model(inputs)
            
            # extract the logits
            logits_x = logits[:num_x]
            logits_u, logits_u_w = logits[num_x:].chunk(2)
            assert logits_u.shape[0] == num_u and logits_u_w.shape[0] == num_u
            del logits
            
            # Supervised loss
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            
            # Semi-supervised loss
            pseudo_label = torch.softmax(logits_u.detach()/temp, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(threshold).float()
            
            total_sum = mask.sum().item()
            
            if total_sum > 0.5: 
                Lu = (F.cross_entropy(logits_u_w, targets_u, reduction='none')*mask).sum() / total_sum

            else:
                Lu = 0
            
            # total loss
            lambda_u = 1.0
            loss = Lx + lambda_u * Lu            
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        LOGGER.info(f'Semi-supervised Training Epoch: [{epoch+1}/{num_epochs}]')
    
    
    # Triplet Loss Fine-tune
    print("Start Triplet Fine-Tuning......")
    model.classify = False
    params_1x = [param for name, param in model.named_parameters() if name not in ["logits.weight", "logits.bias"]]
    grouped_parameters = [
        {'params': params_1x, 'lr': learning_rate*20},
        {'params': model.logits.parameters(), 'lr': 0}
    ]
    optimizer = torch.optim.SGD(grouped_parameters, lr=learning_rate, weight_decay=0.001)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_epochs*num_iters)
    
    triplet_loss = TripletMarginLoss(margin=1, p=2)
    device = get_device()
    model.classify = False
    model.eval()
    for epoch in range(20):
        labeled_iter = iter(labeled_trainloader)
        all_embeddings = None
        all_ys = None
        for X, y in labeled_iter:
            X = X.to(device)
            embedding = model(X)
            if all_embeddings is None:
                all_embeddings = embedding
                all_ys = y
            else:
                all_embeddings = torch.cat((all_embeddings, embedding))
                all_ys = torch.cat((all_ys, y))
        # Form the triplets: (anchor, positive, hard negative)
        anchor, positive, negative = None, None, None
        for idx, (e, y) in enumerate(zip(all_embeddings, all_ys)):
            min_neg_dist, min_neg = 999999, None
            pair_counter = 0
            for idx1, (e1, y1) in enumerate(zip(all_embeddings, all_ys)):
                if idx == idx1:
                    continue
                # Include all anchor-positive pairs
                if y == y1:
                    pair_counter += 1
                    if anchor is None:
                        anchor, positive = e.unsqueeze(0), e1.unsqueeze(0)
                    else:
                        anchor, positive = torch.cat((anchor, e.unsqueeze(0))), torch.cat((positive, e1.unsqueeze(0)))
                else:
                    curr_dist = (e-e1).norm().item()
                    if curr_dist < min_neg_dist:
                        min_neg_dist = curr_dist
                        min_neg = e1
            if negative is None:
                negative = torch.cat(tuple([min_neg.unsqueeze(0) for _ in range(pair_counter)]))
            else:
                negative = torch.cat((negative, torch.cat(tuple([min_neg.unsqueeze(0) for _ in range(pair_counter)]))))
        loss = triplet_loss(anchor, positive, negative)
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        LOGGER.info(f'Triplet Training Epoch: [{epoch+1}/{20}]')
        


def train(source, face_folder):
    '''
    train():
        This function is used to tune the pretrained face recognition network using the semi-supervised learning + triplet loss
    inputs:
        source: path to the video source
        face_folder: path to the face labeled set
    '''
    '''
    NOTE: All images in the training set are in the RGB format!
    '''
    # Load small labeled dataset provided by the user
    LOGGER.info("Loading the labeled dataset......")
    LabeledTrainSet = TrainingSetLabeled(face_folder)
    LabeledTrainLoader = DataLoader(LabeledTrainSet, batch_size=8, shuffle=True)
    # Load unlabeled dataset obtained from the video
    LOGGER.info("Loading the unlabeled dataset......")
    UnlabeledTrainSet = TrainingSetUnlabeled(source, "data", num_images=2048)
    UnlabeledTrainLoader = DataLoader(UnlabeledTrainSet, batch_size=8*10, shuffle=True)
    # Get the number of characters of interest
    num_classes = LabeledTrainSet.get_num_classes()
    # Load pretrained face recognition network
    fr_model = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=num_classes
    ).to(get_device())
    # Initialize the weight-parameters of the classification 
    nn.init.xavier_uniform_(fr_model.logits.weight)
    # Tune the face recognition network using semi-supervised training + triplet loss
    LOGGER.info("Start training the recognition network......")
    train_SSL(fr_model, LabeledTrainLoader, UnlabeledTrainLoader, num_epochs=10, num_iters=40, learning_rate=1.5e-4, threshold=0.99)
    # Return the embedding pool for the labeled images, which is used to match faces according to embedding similarities
    fr_model.classify = False
    LabeledTrainSet = TrainingSetLabeled(face_folder, transform=False)
    LabeledTrainLoader = DataLoader(LabeledTrainSet, batch_size=8, shuffle=False)
    embedding_pool = EmbeddingPool(LabeledTrainLoader, fr_model, get_device(), threshold=0.8)
    return fr_model, embedding_pool, num_classes