import torch
class EmbeddingPool:
    def __init__(self, DataLoader, resnet, device, threshold=1.21, threshold_high=1.35):
        self.threshold = threshold
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
        # resnet.classify = True

    def compare(self, embedding):
        min_dist = 999999
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
        min_dist = 999999
        for yi in self.embedding_dict:
            curr_dist = torch.mean(torch.tensor([(e1-embedding).norm().item() for e1 in self.embedding_dict[yi]])).item()
            if curr_dist < min_dist:
                min_dist = curr_dist
        return min_dist
