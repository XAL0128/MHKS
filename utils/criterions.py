import torch
from torch import nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, pred, target_is_real=True):
        target_tensor = torch.tensor(1 if target_is_real else 0, dtype=torch.long).to(pred.device)
        loss = self.loss(pred, target_tensor.expand(pred.size(0), ))

        return {'loss': loss, 'pred': torch.max(pred, dim=1)[1]}


class EstLoss(nn.Module):
    def __init__(self, config):
        super(EstLoss, self).__init__()
        self.radius = config.emb_radius

    def forward(self, mode, **kwargs):
        assert mode in ['main', 'est']
        if mode == 'main':
            # (1) Density estimation
            loss_est = -kwargs['output'].mean()
            # (2) Making embedding located in [-radius, radius].
            emb = torch.cat(kwargs['emb'], dim=0)
            loss_wall = torch.relu(torch.abs(emb) - self.radius).square().mean()
            # Return
            return {'loss_est': loss_est, 'loss_wall': loss_wall}
        else:
            # (1) Real & fake losses
            loss_real = torch.mean((1.0 - kwargs['output_real']) ** 2)
            loss_fake = torch.mean((1.0 + kwargs['output_fake']) ** 2)
            # (2) Making outputs of the estimator to be zero-centric
            outputs = torch.cat([kwargs['output_real'], kwargs['output_fake']], dim=0)
            loss_zc = torch.mean(outputs).square()
            # Return
            return {'loss_real': loss_real, 'loss_fake': loss_fake, 'loss_zc': loss_zc}


class ContrastLoss(nn.Module):
    def __init__(self, config):
        super(ContrastLoss, self).__init__()
        self.criterions = nn.CrossEntropyLoss()
        self.align_temperature = config.align_temperature
        self.intra_temperature = config.intra_temperature
        self.dis_temperature = config.dis_temperature

    def forward(self, mode='align', **kwargs):
        if mode == 'align':
            loss = self.align_loss(kwargs['anchor'], kwargs['sample'])
        elif mode == 'intra':
            loss = self.intra_loss(kwargs['data'], kwargs['label'])
        elif mode == 'disentangle':
            loss = self.disentangled_loss(kwargs['relevant'], kwargs['irrelevant'])
        else:
            raise ValueError("Expected ‘align’ or ‘intra’ or ‘disentangle’, but received {}.".format(mode))

        return loss

    def align_loss(self, anchor, sample):
        batch_size = anchor.shape[0]

        labels = torch.arange(batch_size, dtype=torch.long, device=anchor.device)

        anchor = F.normalize(anchor, dim=-1, p=2)  # (batch size, dim) -> (50, 150)
        sample = F.normalize(sample, dim=-1, p=2)

        logits1 = anchor @ sample.t()
        logits2 = sample @ anchor.t()

        loss1 = self.criterions(logits1 / self.align_temperature, labels)
        loss2 = self.criterions(logits2 / self.align_temperature, labels)

        loss = loss1 + loss2

        return loss

    def intra_loss(self, x, label):
        y = label.view(-1)
        positive_index = (y > 0).nonzero().view(-1)
        negative_index = (y < 0).nonzero().view(-1)

        x_p = F.normalize(x[positive_index], dim=-1, p=2)
        x_n = F.normalize(x[negative_index], dim=-1, p=2)

        positive_similarity_matrix = F.cosine_similarity(x_p.unsqueeze(1), x_p.unsqueeze(0), dim=2)
        negative_similarity_matrix = F.cosine_similarity(x_p.unsqueeze(1), x_n.unsqueeze(0), dim=2)

        mask = torch.eye(x_p.shape[0], dtype=torch.bool).cuda()
        mask = ~mask
        positive_pair = mask * positive_similarity_matrix

        nominator = torch.exp(positive_pair.mean() / self.intra_temperature)
        denominator = torch.exp(positive_pair.mean() / self.intra_temperature) + \
                      torch.exp(negative_similarity_matrix.mean() / self.intra_temperature)

        loss = - torch.log(nominator / denominator)

        return loss

    def disentangled_loss(self, useful, irrelevant):
        positive_data = F.normalize(useful, dim=-1, p=2)
        negative_data = F.normalize(irrelevant, dim=-1, p=2)

        positive_similarity_matrix = F.cosine_similarity(positive_data.unsqueeze(1), positive_data.unsqueeze(0), dim=2)
        negative_similarity_matrix = F.cosine_similarity(positive_data.unsqueeze(1), negative_data.unsqueeze(0), dim=2)

        mask = torch.eye(positive_data.shape[0], dtype=torch.bool).cuda()
        mask = ~mask
        positive_pair = mask * positive_similarity_matrix

        nominator = torch.exp(positive_pair.mean() / self.dis_temperature)
        denominator = torch.exp(positive_pair.mean() / self.dis_temperature) + \
                      torch.exp(negative_similarity_matrix.mean() / self.dis_temperature)

        loss = - torch.log(nominator / denominator)

        return loss
