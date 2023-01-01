import torch
class EmbeddingPool:
    def __init__(self, DataLoader, UnlabeledTrainLoader, resnet, device, threshold_high=1.5):
        resnet.classify = False
        self.threshold_high = threshold_high
        all_embeddings = None
        all_ys = None
        with torch.no_grad(): 
            # Obtain all the embeddings  
            for X, y in DataLoader:
                X = X.to(device)
                embedding = resnet(X)
                if all_embeddings is None:
                    all_embeddings = embedding
                    all_ys = y
                else:
                    all_embeddings = torch.cat((all_embeddings, embedding))
                    all_ys = torch.cat((all_ys, y))
        self.embedding_dict = {}
        for ei, yi in zip(all_embeddings, all_ys):
            yi = yi.item()
            if not yi in self.embedding_dict:
                self.embedding_dict[yi] = ei.unsqueeze(0)
            else:
                self.embedding_dict[yi] = torch.cat((self.embedding_dict[yi], ei.unsqueeze(0)))
        if UnlabeledTrainLoader is not None:
            self.threshold = self.adaptive_threshold(UnlabeledTrainLoader, resnet, device)
            print(f'Threshold value is {self.threshold}')
        else:
            self.threshold = 0.85
        # resnet.classify = True
        
    def adaptive_threshold(self, UnlabeledTrainLoader, resnet, device):
        resnet.eval()
        dists = []
        with torch.no_grad():
            for _, X  in UnlabeledTrainLoader:
                resnet.classify = True
                X = X.to(device)
                logits = resnet(X)
                pseudo_label = torch.softmax(logits.detach(), dim=-1)
                max_prob, y_hat = torch.max(pseudo_label, axis=1)
                max_prob = max_prob.cpu().numpy()
                
                resnet.classify = False
                embeddings = resnet(X)
                for prob, embedding in zip(max_prob, embeddings):
                    if prob > 0.995:
                        dist = self.get_min_dist(embedding)
                        dists.append(dist)
        dists.sort()
        thresh = dists[int(len(dists)*0.8)]   
        if thresh > 1.05:
            thresh = 1.05
        elif thresh < 0.85:
            thresh = 0.85
        return thresh              
                
    def compare(self, embedding):
        min_dist = float('inf')
        min_class = None
        for yi in self.embedding_dict:
            curr_dist = torch.mean(torch.tensor([(e1-embedding).norm().item() for e1 in self.embedding_dict[yi]])).item()
            if curr_dist < min_dist:
                min_dist = curr_dist
                min_class = yi
        if min_dist > self.threshold:
            if min_dist > self.threshold_high:
                min_class = -1 # unseen person
            else: 
                min_class = -2 # not sure: unseen person or unclear person
        return min_class

    def get_min_dist(self, embedding):
        min_dist = float('inf')
        for yi in self.embedding_dict:
            curr_dist = torch.mean(torch.tensor([(e1-embedding).norm().item() for e1 in self.embedding_dict[yi]])).item()
            if curr_dist < min_dist:
                min_dist = curr_dist
        return min_dist