import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.init import xavier_normal


class Addition(nn.Module):
    def __init__(self):
        super(Addition, self).__init__()

    def forward(self, l, a, v):
        return l + a + v


class Multiplication(nn.Module):
    def __init__(self):
        super(Multiplication, self).__init__()

    def forward(self, l, a, v):
        return l * a * v


class Concat(nn.Module):
    def __init__(self, config):
        super(Concat, self).__init__()
        # parameters
        self.d_l = config.latent_dim

        # concat fusion layer
        self.concat_layer = nn.Sequential(
            nn.Linear(self.d_l * 3, self.d_l),
            nn.ReLU()
        )

    def forward(self, l, a, v):
        concat_f = torch.cat([l, a, v], dim=-1)
        return self.concat_layer(concat_f)


class TensorFusion(nn.Module):
    def __init__(self, config):
        super(TensorFusion, self).__init__()
        self.d_l = config.latent_dim
        self.dropout = config.dropout
        self.hidden = 64
        self.DTYPE = torch.cuda.FloatTensor

        # tensor fusion layer
        self.tensor_layer = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear((self.d_l + 1) * (self.d_l + 1) * (self.d_l + 1), self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.d_l),
            nn.Tanh()
        )

    def forward(self, l, a, v):
        _audio_h = torch.cat((Variable(torch.ones(a.size(0), 1).type(self.DTYPE), requires_grad=False), a), dim=1)
        _video_h = torch.cat((Variable(torch.ones(a.size(0), 1).type(self.DTYPE), requires_grad=False), v), dim=1)
        _text_h = torch.cat((Variable(torch.ones(a.size(0), 1).type(self.DTYPE), requires_grad=False), l), dim=1)

        fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))
        fusion_tensor = fusion_tensor.view(-1, (a.size(1) + 1) * (a.size(1) + 1), 1)
        fusion_tensor = torch.bmm(fusion_tensor, _text_h.unsqueeze(1)).view(a.size(0), -1)

        output = self.tensor_layer(fusion_tensor)

        return output


class Low_Rank_Fusion(nn.Module):
    def __init__(self, config):
        super(Low_Rank_Fusion, self).__init__()
        self.d_l = config.latent_dim
        self.dropout = config.dropout
        self.hidden = 64
        self.DTYPE = torch.cuda.FloatTensor
        self.rank = 5

        # low_rank fusion layer
        self.audio_factor = Parameter(torch.Tensor(self.rank, self.d_l + 1, self.d_l).cuda())
        self.video_factor = Parameter(torch.Tensor(self.rank, self.d_l + 1, self.d_l).cuda())
        self.text_factor = Parameter(torch.Tensor(self.rank, self.d_l + 1, self.d_l).cuda())
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank).cuda())
        self.fusion_bias = Parameter(torch.Tensor(1, self.d_l).cuda())

        xavier_normal(self.audio_factor)
        xavier_normal(self.video_factor)
        xavier_normal(self.text_factor)
        xavier_normal(self.fusion_weights)
        xavier_normal(self.fusion_bias)

    def forward(self, l, a, v):
        _audio_h = torch.cat((Variable(torch.ones(a.size(0), 1).type(self.DTYPE), requires_grad=False), a), dim=1)
        _video_h = torch.cat((Variable(torch.ones(a.size(0), 1).type(self.DTYPE), requires_grad=False), v), dim=1)
        _text_h = torch.cat((Variable(torch.ones(a.size(0), 1).type(self.DTYPE), requires_grad=False), l), dim=1)

        fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        fusion_video = torch.matmul(_video_h, self.video_factor)
        fusion_text = torch.matmul(_text_h, self.text_factor)
        fusion_zy = fusion_audio * fusion_video * fusion_text

        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        y_2 = output.view(-1, self.d_l)

        return y_2


class GraphFusion(nn.Module):
    def __init__(self, config):
        super(GraphFusion, self).__init__()
        self.d_l = config.latent_dim
        self.dropout = config.dropout
        self.hidden = 64

        # graph layer
        self.graph_fusion = nn.Sequential(
            nn.Linear(self.d_l * 2, self.hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hidden, self.d_l),
            nn.Tanh()
        )

        self.graph_fusion2 = nn.Sequential(
            nn.Linear(self.d_l * 2, self.hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hidden, self.d_l),
            nn.Tanh()
        )

        self.attention = nn.Linear(self.d_l, 1)
        self.linear_layer = nn.Sequential(
            nn.Linear(self.d_l * 3, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.d_l),
            nn.Tanh()
        )

    def forward(self, l, a, v):
        ###################### unimodal layer  ##########################
        sa = torch.tanh(self.attention(a))
        sv = torch.tanh(self.attention(v))
        sl = torch.tanh(self.attention(l))

        total_weights = torch.cat([sa, sv, sl], 1)

        unimodal_a = (sa.expand(a.size(0), self.d_l))
        unimodal_v = (sv.expand(a.size(0), self.d_l))
        unimodal_l = (sl.expand(a.size(0), self.d_l))
        sa = sa.squeeze(1)
        sl = sl.squeeze(1)
        sv = sv.squeeze(1)
        unimodal = (unimodal_a * a + unimodal_v * v + unimodal_l * l) / 3

        ##################### bimodal layer ############################
        a = F.softmax(a, 1)
        v = F.softmax(v, 1)
        l = F.softmax(l, 1)
        sav = (1 / (torch.matmul(a.unsqueeze(1), v.unsqueeze(2)).squeeze() + 0.5) * (sa + sv))
        sal = (1 / (torch.matmul(a.unsqueeze(1), l.unsqueeze(2)).squeeze() + 0.5) * (sa + sl))
        svl = (1 / (torch.matmul(v.unsqueeze(1), l.unsqueeze(2)).squeeze() + 0.5) * (sl + sv))

        normalize = torch.cat([sav.unsqueeze(1), sal.unsqueeze(1), svl.unsqueeze(1)], 1)
        normalize = F.softmax(normalize, 1)
        total_weights = torch.cat([total_weights, normalize], 1)

        a_v = torch.tanh(
            (normalize[:, 0].unsqueeze(1).expand(a.size(0), self.d_l)) * self.graph_fusion(torch.cat([a, v], 1)))
        a_l = torch.tanh(
            (normalize[:, 1].unsqueeze(1).expand(a.size(0), self.d_l)) * self.graph_fusion(torch.cat([a, l], 1)))
        v_l = torch.tanh(
            (normalize[:, 2].unsqueeze(1).expand(a.size(0), self.d_l)) * self.graph_fusion(torch.cat([v, l], 1)))
        bimodal = (a_v + a_l + v_l)

        ###################### trimodal layer ####################################
        a_v2 = F.softmax(self.graph_fusion(torch.cat([a, v], 1)), 1)
        a_l2 = F.softmax(self.graph_fusion(torch.cat([a, l], 1)), 1)
        v_l2 = F.softmax(self.graph_fusion(torch.cat([v, l], 1)), 1)
        savvl = (1 / (torch.matmul(a_v2.unsqueeze(1), v_l2.unsqueeze(2)).squeeze() + 0.5) * (sav + svl))
        saavl = (1 / (torch.matmul(a_v2.unsqueeze(1), a_l2.unsqueeze(2)).squeeze() + 0.5) * (sav + sal))
        savll = (1 / (torch.matmul(a_l2.unsqueeze(1), v_l2.unsqueeze(2)).squeeze() + 0.5) * (sal + svl))
        savl = (1 / (torch.matmul(a_v2.unsqueeze(1), l.unsqueeze(2)).squeeze() + 0.5) * (sav + sl))
        salv = (1 / (torch.matmul(a_l2.unsqueeze(1), v.unsqueeze(2)).squeeze() + 0.5) * (sal + sv))
        svla = (1 / (torch.matmul(v_l2.unsqueeze(1), a.unsqueeze(2)).squeeze() + 0.5) * (sa + svl))

        normalize2 = torch.cat(
            [savvl.unsqueeze(1), saavl.unsqueeze(1), savll.unsqueeze(1), savl.unsqueeze(1), salv.unsqueeze(1),
             svla.unsqueeze(1)], 1)
        normalize2 = F.softmax(normalize2, 1)
        total_weights = torch.cat([total_weights, normalize2], 1)

        avvl = torch.tanh((normalize2[:, 0].unsqueeze(1).expand(a.size(0), self.d_l)) * self.graph_fusion2(
            torch.cat([a_v, v_l], 1)))
        aavl = torch.tanh((normalize2[:, 1].unsqueeze(1).expand(a.size(0), self.d_l)) * self.graph_fusion2(
            torch.cat([a_v, a_l], 1)))
        avll = torch.tanh((normalize2[:, 2].unsqueeze(1).expand(a.size(0), self.d_l)) * self.graph_fusion2(
            torch.cat([v_l, a_l], 1)))
        avl = torch.tanh((normalize2[:, 3].unsqueeze(1).expand(a.size(0), self.d_l)) * self.graph_fusion2(
            torch.cat([a_v, l], 1)))
        alv = torch.tanh((normalize2[:, 4].unsqueeze(1).expand(a.size(0), self.d_l)) * self.graph_fusion2(
            torch.cat([a_l, v], 1)))
        vla = torch.tanh((normalize2[:, 5].unsqueeze(1).expand(a.size(0), self.d_l)) * self.graph_fusion2(
            torch.cat([v_l, a], 1)))
        trimodal = (avvl + aavl + avll + avl + alv + vla)
        fusion = torch.cat([unimodal, bimodal, trimodal], 1)

        output = self.linear_layer(fusion)

        return output


class Fusion(nn.Module):
    def __init__(self, config):
        super(Fusion, self).__init__()
        # parameters
        self.mode = config.fusion_mode

        if self.mode == 'add':
            self.fusion_layer = Addition()
        elif self.mode == 'concat':
            self.fusion_layer = Concat(config)
        elif self.mode == 'multiple':
            self.fusion_layer = Multiplication()
        elif self.mode == 'tensor':
            self.fusion_layer = TensorFusion(config)
        elif self.mode == 'low_rank':
            self.fusion_layer = Low_Rank_Fusion(config)
        elif self.mode == 'graph':
            self.fusion_layer = GraphFusion(config)
        else:
            # print('Default mode is concat')
            self.fusion_layer = Concat(config)

    def forward(self, l, a, v):
        """
        :param l: text representation
        :param a: audio representation
        :param v: video representation
        :param mode: multimodal fusion method, choice=['add', 'concat', 'multiple', 'low_rank', 'tensor', 'graph']
        :return: joint representation for prediction
        """

        fusion_output = self.fusion_layer(l, a, v)

        return fusion_output