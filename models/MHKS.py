import os.path

import torch
import torch.nn as nn
from models.base_models import L_Encoder, A_Encoder, V_Encoder, Disen_Encoder, Regression_Decoder, DensityEstimator, Reconstructor, Disriminator, Classifier
from models.fusion_method import Fusion


class MHKS(nn.Module):
    def __init__(self, config):
        super(MHKS, self).__init__()
        self.extractor = Extractor(config)
        self.encoder_r = Enc_relevant(config)
        self.encoder_ir = Enc_irrelevant(config)
        self.fusion = Fusion(config)
        self.decoder = DisenIB_Decoder(config)
        self.reconstructor = DisenIB_Reconstructor(config)
        self.estimator = DisenIB_Density(config)
        self.classifier = Classifier(config)
        self.disriminator = DisenIB_Discriminator(config)

    def forward(self, input_ids, visual, acoustic, input_mask, segment_ids, label_ids):
        # extract feature
        feature_map = self.extractor(input_ids, visual, acoustic, input_mask, segment_ids)
        # encoder
        feature_r = self.encoder_r(feature_map)
        feature_ir = self.encoder_ir(feature_map)
        # fusion + classify
        fusion_f = self.fusion(feature_r['l'], feature_r['a'], feature_r['v'])
        task_output = self.classifier(fusion_f)
        # decoder
        decode_output = self.decoder(feature_r)
        # reconstructor
        recon_output = self.reconstructor(feature_ir, label_ids)
        # estimator
        est_output = self.estimator(feature_r, feature_ir, mode='orig')
        # discriminator
        disc_output = self.disriminator(recon_output)

        results = {
            'feature_map': feature_map,
            'feature_r': feature_r,
            'feature_ir': feature_ir,
            'fusion_f': fusion_f,
            'task_output': task_output,
            'decode_output': decode_output,
            'recon_output': recon_output,
            'est_output': est_output,
            'disc_output': disc_output
        }

        return results


class Extractor(nn.Module):
    def __init__(self, config):
        super(Extractor, self).__init__()
        self.d_l = config.latent_dim

        self.l_encoder = L_Encoder.from_pretrained(config.pretrained, num_labels=1)
        self.proj_l = nn.Conv1d(config.TEXT_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)
        self.a_encoder = A_Encoder(config)
        self.v_encoder = V_Encoder(config)

    def forward(self, input_ids, visual, acoustic, input_mask, segment_ids):
        output_l = self.l_encoder(
            input_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask)
        output_l = self.proj_l(output_l)
        output_a = self.a_encoder(acoustic)
        output_v = self.v_encoder(visual)

        return {'l': output_l[:, :, -1], 'a': output_a[-1], 'v': output_v[-1]}


class Enc_relevant(nn.Module):
    def __init__(self, config):
        super(Enc_relevant, self).__init__()
        self.lc_encoder = Disen_Encoder(config)
        self.ac_encoder = Disen_Encoder(config)
        self.vc_encoder = Disen_Encoder(config)

    def forward(self, feature):
        lr = self.lc_encoder(feature['l'])
        ar = self.ac_encoder(feature['a'])
        vr = self.vc_encoder(feature['v'])

        return {'l': lr, 'a': ar, 'v': vr}


class Enc_irrelevant(nn.Module):
    def __init__(self, config):
        super(Enc_irrelevant, self).__init__()
        self.input_size = config.latent_dim

        self.ls_encoder = Disen_Encoder(config)
        self.as_encoder = Disen_Encoder(config)
        self.vs_encoder = Disen_Encoder(config)

    def forward(self, feature):
        li = self.ls_encoder(feature['l'])
        ai = self.as_encoder(feature['a'])
        vi = self.vs_encoder(feature['v'])

        return {'l': li, 'a': ai, 'v': vi}


class DisenIB_Decoder(nn.Module):
    def __init__(self, config):
        super(DisenIB_Decoder, self).__init__()
        self.r_std = config.relevant_std

        self.lc_decoder = Regression_Decoder(config)
        self.ac_decoder = Regression_Decoder(config)
        self.vc_decoder = Regression_Decoder(config)

    def resampling(self, mu, std):
        eps = torch.randn(mu.size(), device=mu.device)
        return eps.mul(std).add(mu)

    def forward(self, encode):
        # decode
        regress_lr = self.lc_decoder(self.resampling(encode['l'], self.r_std))
        regress_ar = self.lc_decoder(self.resampling(encode['a'], self.r_std))
        regress_vr = self.lc_decoder(self.resampling(encode['v'], self.r_std))

        return {'l': regress_lr, 'a': regress_ar, 'v': regress_vr}


class Prediction_head(nn.Module):
    def __init__(self, config):
        super(Prediction_head, self).__init__()
        self.r_std = config.relevant_std

        self.classifier = Classifier(config)

    def resampling(self, mu, std):
        eps = torch.randn(mu.size(), device=mu.device)
        return eps.mul(std).add(mu)

    def forward(self, fusion):
        output = self.classifier(self.resampling(fusion, self.r_std))

        return output


class DisenIB_Reconstructor(nn.Module):
    def __init__(self, config):
        super(DisenIB_Reconstructor, self).__init__()
        self.ir_std = config.irrelevant_std

        self.lc_decoder = Reconstructor(config)
        self.ac_decoder = Reconstructor(config)
        self.vc_decoder = Reconstructor(config)

    def resampling(self, mu, std):
        eps = torch.randn(mu.size(), device=mu.device)
        return eps.mul(std).add(mu)

    def forward(self, encode, label):
        # reconstruct
        recon_lr = self.lc_decoder(self.resampling(encode['l'], self.ir_std), label)
        recon_ar = self.lc_decoder(self.resampling(encode['a'], self.ir_std), label)
        recon_vr = self.lc_decoder(self.resampling(encode['v'], self.ir_std), label)

        return {'l': recon_lr, 'a': recon_ar, 'v': recon_vr}


class DisenIB_Density(nn.Module):
    def __init__(self, config):
        super(DisenIB_Density, self).__init__()
        self.r_std = config.est_r_std
        self.ir_std = config.est_ir_std

        self.l_estimator = DensityEstimator(config)
        self.a_estimator = DensityEstimator(config)
        self.v_estimator = DensityEstimator(config)

    def resampling(self, mu, std):
        eps = torch.randn(mu.size(), device=mu.device)
        return eps.mul(std).add(mu)

    def forward(self, relevant, irrelevant, mode):
        est_l = self.l_estimator(self.resampling(relevant['l'], self.r_std),
                                 self.resampling(irrelevant['l'], self.ir_std), mode=mode)
        est_a = self.a_estimator(self.resampling(relevant['a'], self.r_std),
                                 self.resampling(irrelevant['a'], self.ir_std), mode=mode)
        est_v = self.v_estimator(self.resampling(relevant['v'], self.r_std),
                                 self.resampling(irrelevant['v'], self.ir_std), mode=mode)

        return {'l': est_l, 'a': est_a, 'v': est_v}


class DisenIB_Discriminator(nn.Module):
    def __init__(self, config):
        super(DisenIB_Discriminator, self).__init__()

        self.l_discriminator = Disriminator(config)
        self.a_discriminator = Disriminator(config)
        self.v_discriminator = Disriminator(config)

    def forward(self, rec_output):
        disc_l = self.l_discriminator(rec_output['l'])
        disc_a = self.a_discriminator(rec_output['a'])
        disc_v = self.v_discriminator(rec_output['v'])

        return {'l': disc_l, 'a': disc_a, 'v': disc_v}
