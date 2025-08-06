import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from modules.transformer import TransformerEncoder
from models.bert import BertModel
from utils.utils import init_weights


# text feature extractor
class L_Encoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return outputs


# audio feature extractor
class A_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj_a = nn.Conv1d(config.ACOUSTIC_DIM, config.latent_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.transa = TransformerEncoder(embed_dim=config.latent_dim,
                                         num_heads=config.num_heads,  # self.num_heads,
                                         layers=config.layers,  # max(self.layers, layers),
                                         attn_dropout=config.attn_dropout,
                                         relu_dropout=config.relu_dropout,  # self.relu_dropout,
                                         res_dropout=config.res_dropout,  # self.res_dropout,
                                         embed_dropout=config.embed_dropout,  # self.embed_dropout,
                                         attn_mask=config.attn_mask)  # self.attn_mask)

    def forward(self, acoustic):
        acoustic = self.proj_a(acoustic.transpose(1, 2))
        acoustic = acoustic.permute(2, 0, 1)
        output = self.transa(acoustic)
        # return output[-1]
        return output


# video feature extractor
class V_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj_a = nn.Conv1d(config.VISUAL_DIM, config.latent_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.transa = TransformerEncoder(embed_dim=config.latent_dim,
                                         num_heads=config.num_heads,  # self.num_heads,
                                         layers=config.layers,  # max(self.layers, layers),
                                         attn_dropout=config.attn_dropout,
                                         relu_dropout=config.relu_dropout,  # self.relu_dropout,
                                         res_dropout=config.res_dropout,  # self.res_dropout,
                                         embed_dropout=config.embed_dropout,  # self.embed_dropout,
                                         attn_mask=config.attn_mask)  # self.attn_mask)

    def forward(self, visual):
        visual = self.proj_a(visual.transpose(1, 2))
        visual = visual.permute(2, 0, 1)
        output = self.transa(visual)
        # return output[-1]
        return output


class FcBlock(nn.Module):
    """
    in_dim: int
    hidden_dim: list
    out_dim: int
    """
    def __init__(self, in_dim, hidden_dim, out_dim, act=nn.LeakyReLU(), use_dropout=True, dropout=0.3, use_bias=True):
        super(FcBlock, self).__init__()
        # 1. Init
        layers = []
        # 2. Generate each layer
        for index, (ipt, opt) in enumerate(zip([in_dim] + hidden_dim, hidden_dim + [out_dim])):
            layers.append(nn.Linear(ipt, opt, bias=use_bias))
            if index < len(hidden_dim):
                layers.append(act)
                if use_dropout:
                    layers.append(nn.Dropout(dropout))

        self.fc_blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_blocks(x)


# disentanglement layer
class Disen_Encoder(nn.Module):
    def __init__(self, config):
        super(Disen_Encoder, self).__init__()
        self.dim = config.latent_dim
        self.hidden_dims = config.enc_hidden_dims
        self.dropout = config.dropout

        self.output_layer = FcBlock(self.dim,
                                    hidden_dim=self.hidden_dims,
                                    out_dim=self.dim,
                                    act=nn.Tanh(),
                                    use_dropout=True,
                                    dropout=self.dropout)

    def forward(self, x):
        output = self.output_layer(x)

        return output


class Regression_Decoder(nn.Module):
    def __init__(self, config):
        super(Regression_Decoder, self).__init__()
        self.dim = config.latent_dim
        self.hidden_dims = config.dec_hidden_dims
        self.dropout = config.dropout

        self.fc_layers = FcBlock(self.dim,
                                 hidden_dim=self.hidden_dims,
                                 out_dim=1,
                                 act=nn.ReLU(),
                                 use_dropout=True,
                                 dropout=self.dropout)

        # self.apply(init_weights)

    def forward(self, x):
        output = self.fc_layers(x)
        return output


class DensityEstimator(nn.Module):
    """
    Estimating probability density.
    """

    def __init__(self, config):
        super(DensityEstimator, self).__init__()
        self.dim = config.latent_dim * 2
        self.hidden_dims = config.est_hidden_dims

        self._fc_style = nn.Linear(in_features=config.latent_dim, out_features=config.latent_dim, bias=True)
        self._fc_class = nn.Linear(in_features=config.latent_dim, out_features=config.latent_dim, bias=True)
        self._fc_blocks = FcBlock(self.dim,
                                  hidden_dim=self.hidden_dims,
                                  out_dim=1,
                                  act=nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  use_dropout=False)

        # Init weights
        # self.apply(init_weights)

    def _call_method(self, style_emb, class_emb):
        style_emb = self._fc_style(style_emb)
        class_emb = self._fc_class(class_emb)
        return self._fc_blocks(torch.cat([style_emb, class_emb], dim=1))

    def forward(self, style_emb, class_emb, mode):
        assert mode in ['orig', 'perm']
        # 1. q(s, t)
        if mode == 'orig':
            return self._call_method(style_emb, class_emb)
        # 2. q(s)q(t)
        else:
            # Permutation
            style_emb_permed = style_emb[torch.randperm(style_emb.size(0)).to(style_emb.device)]
            class_emb_permed = class_emb[torch.randperm(class_emb.size(0)).to(class_emb.device)]
            return self._call_method(style_emb_permed, class_emb_permed)


class Disriminator(nn.Module):
    def __init__(self, config):
        super(Disriminator, self).__init__()
        self.dim = config.latent_dim
        self.hidden_dims = config.dis_hidden_dims
        self.discriminator = FcBlock(self.dim,
                                     hidden_dim=self.hidden_dims,
                                     out_dim=2,
                                     act=nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                     use_dropout=False)

        # Init weights
        # self.apply(init_weights)

    def forward(self, x):
        output = self.discriminator(x)
        return output


class Reconstructor(nn.Module):
    def __init__(self, config):
        super(Reconstructor, self).__init__()
        self.dim = config.latent_dim
        self.hidden_dims = config.rec_hidden_dims
        self.dropout = config.dropout

        self.feature_dim = nn.Linear(self.dim, self.dim)
        self.label_emb = nn.Linear(1, self.dim)

        self.decoder = FcBlock(self.dim * 2,
                               hidden_dim=self.hidden_dims,
                               out_dim=self.dim,
                               act=nn.ReLU(),
                               use_dropout=True,
                               dropout=self.dropout)

    def forward(self, x, label):
        # get feature embedding and label embedding
        x_emb = F.leaky_relu(self.feature_dim(x), negative_slope=0.2)
        label_emb = F.leaky_relu(self.label_emb(label.squeeze(dim=2)), negative_slope=0.2)
        # concat
        joint = torch.cat((x_emb, label_emb), dim=1)

        output = self.decoder(joint)
        return output


class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_l = config.latent_dim
        self.classifier = nn.Linear(self.d_l, 1)

    def forward(self, x):
        output = self.classifier(x)

        return output



