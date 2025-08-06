import json
import os
import time
import warnings
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
import random
from datetime import datetime
from torch.optim import AdamW
from easydict import EasyDict as edict
from collections import defaultdict
import argparse
from torchprofile import profile_macs

from models.MHKS import MHKS
from utils.criterions import GANLoss, EstLoss, ContrastLoss
from utils.data_preparation import set_up_data_loader
from utils.utils import set_random_seed, dict2detach, set_requires_grad, TriggerPeriod, mkdirs, dict_to_str, set_logger
from utils.metrics import Metrics

parser = argparse.ArgumentParser("Multimodal Hierarchical Knowledge Stripping", add_help=True)
# dataset
parser.add_argument("--dataset_name", type=str, default='sims',
                    choices=['mosi', 'mosei', 'sims'], required=False, help='used dataset name')
parser.add_argument("--max_seq_length", type=int, default=50, required=False)
# base model
parser.add_argument("--irrelevant_std", type=float, default=0.1, required=False, help='')
parser.add_argument("--relevant_std", type=float, default=1.0, required=False, help='')
parser.add_argument("--emb_radius", type=float, default=3.0, required=False, help='')
parser.add_argument("--disc_limit_acc", type=float, default=0.8, required=False)
parser.add_argument("--est_ir_std", type=float, default=0.1, required=False)
parser.add_argument("--est_r_std", type=float, default=0.1, required=False)
parser.add_argument("--est_style_optimize", type=int, default=4, required=False)
parser.add_argument("--dropout", type=float, default=0.2, required=False)
# loss weight
parser.add_argument("--lambda_task", type=float, default=12.0, required=False)
parser.add_argument("--lambda_dec", type=float, default=1.5, required=False)
parser.add_argument("--lambda_rec", type=float, default=1.5, required=False)
parser.add_argument("--lambda_est", type=float, default=0.5, required=False)
parser.add_argument("--lambda_est_zc", type=float, default=0.05, required=False)
parser.add_argument("--lambda_wall", type=float, default=10.0, required=False)
parser.add_argument("--lambda_disc", type=float, default=0.1, required=False)
parser.add_argument("--lambda_cl_align", type=float, default=0.5, required=False)
parser.add_argument("--lambda_cl_intra", type=float, default=5.0, required=False)
parser.add_argument("--lambda_cl_disentangle", type=float, default=5.0, required=False)
parser.add_argument("--align_temperature", type=float, default=0.8, required=False)
parser.add_argument("--intra_temperature", type=float, default=0.4, required=False)
parser.add_argument("--dis_temperature", type=float, default=0.6, required=False)
# transformer param
parser.add_argument("--num_heads", type=int, default=10, required=False)
parser.add_argument("--layers", type=int, default=5, required=False)
parser.add_argument("--attn_dropout", type=float, default=0.3, required=False)
parser.add_argument("--relu_dropout", type=float, default=0.0, required=False)
parser.add_argument("--res_dropout", type=float, default=0.0, required=False)
parser.add_argument("--embed_dropout", type=float, default=0.2, required=False)
parser.add_argument("--attn_mask", type=bool, default=False, required=False)
# Epochs & batch size & optimization
parser.add_argument("--n_epochs", type=int, default=40, required=False)
parser.add_argument("--latent_dim", type=int, default=100, required=False, help='latent feature dimension')
parser.add_argument("--fusion_mode", type=str, default='concat',
                    choices=['add', 'concat', 'multiple', 'low_rank', 'tensor', 'graph'],
                    required=False, help='multimodal fusion method')
parser.add_argument("--batch_size", type=int, default=32, required=False)
parser.add_argument("--other_lr", type=float, default=4e-5, required=False)
parser.add_argument("--main_lr", type=float, default=2e-5, required=False)
parser.add_argument("--weight_decay", type=float, default=0.003, required=False)
# model structure param
parser.add_argument("--enc_hidden_dims", type=list, default=[64, 64], required=False)
parser.add_argument("--dec_hidden_dims", type=list, default=[128, 64, 32], required=False)
parser.add_argument("--est_hidden_dims", type=list, default=[256, 256], required=False)
parser.add_argument("--dis_hidden_dims", type=list, default=[64, 64], required=False)
parser.add_argument("--rec_hidden_dims", type=list, default=[256, 256], required=False)
# others
parser.add_argument("--device", type=str, default='2', required=False)
parser.add_argument("--log", type=str, default='logs',
                    required=False)
parser.add_argument("--result", type=str, default='results',
                    required=False)
parser.add_argument("--csv_file", type=str, default='sims_test.csv',
                    required=False)
parser.add_argument("--seed", type=int, default=312, required=False)

args = parser.parse_args()
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = 'cuda'

dataset_param = {
    'mosi': {
        'TEXT_DIM': 768,
        'ACOUSTIC_DIM': 74,
        'VISUAL_DIM': 47,
        'data_path': 'datasets/mosi.pkl',
        'pretrained': 'bert-base-uncased',
        'task_loss': 'MAE'
    },
    'mosei': {
        'TEXT_DIM': 768,
        'ACOUSTIC_DIM': 74,
        'VISUAL_DIM': 35,
        'data_path': 'datasets/mosei.pkl',
        'pretrained': 'bert-base-uncased',
        'task_loss': 'MSE'
    },
    'sims': {
        'TEXT_DIM': 768,
        'ACOUSTIC_DIM': 33,
        'VISUAL_DIM': 709,
        'data_path': 'datasets/sims.pkl',
        'pretrained': 'bert-base-chinese',
        'task_loss': 'MAE'
    }
}


class Train:
    def __init__(self, dataloader, config, log):
        # parameter & tools
        self.dataset_name = config.dataset_name
        self.config = config
        self.logger = log
        self.metrics = Metrics().getMetics(self.dataset_name)
        self.meters = TriggerPeriod(period=self.config.est_style_optimize + 1, area=self.config.est_style_optimize)
        # dataloader
        self.dataloader = dataloader
        # network
        self.model = MHKS(config).to(device)

        # criterion
        if config.task_loss == 'MAE':
            self.loss_task = nn.L1Loss().to(device)
        else:
            self.loss_task = nn.MSELoss().to(device)
        self.loss_recon = nn.MSELoss().to(device)
        self.loss_disc = GANLoss().to(device)
        self.loss_est = EstLoss(config).to(device)
        self.loss_cl = ContrastLoss(config).to(device)

        # main architecture optimizer
        self.optimizers_main = AdamW(
            list(self.model.extractor.parameters()) +
            list(self.model.encoder_r.parameters()) +
            list(self.model.encoder_ir.parameters()) +
            list(self.model.fusion.parameters()) +
            list(self.model.decoder.parameters()) +
            list(self.model.classifier.parameters()) +
            list(self.model.reconstructor.parameters()),
            lr=self.config.main_lr,
            weight_decay=self.config.weight_decay)
        # discriminator optimizer
        self.optimizers_disc = AdamW(self.model.disriminator.parameters(), lr=config.other_lr,
                                     weight_decay=self.config.weight_decay)
        # density estimator optimizer
        self.optimizers_est = AdamW(self.model.estimator.parameters(), lr=config.other_lr,
                                    weight_decay=self.config.weight_decay)

    def train_one_epoch(self):
        self.model.train()
        ################################################################################################################
        # main architecture optimization
        ################################################################################################################
        tr_loss = 0
        nb_tr_steps = 0
        for step, batch in enumerate(self.dataloader['train']):
            # load data
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = self.parse_batch(batch)

            # set optimizer
            set_requires_grad([self.model.extractor, self.model.encoder_r, self.model.encoder_ir,
                               self.model.fusion, self.model.decoder, self.model.reconstructor,
                               self.model.classifier], requires_grad=True)
            set_requires_grad([self.model.estimator, self.model.disriminator], requires_grad=False)
            self.optimizers_main.zero_grad()

            results = self.model(input_ids, visual, acoustic, input_mask, segment_ids, label_ids)
            loss_dict = self.get_total_loss(results, label_ids)

            # main
            loss1 = loss_dict['task'] + loss_dict['decode'] + loss_dict['reconstruct'] + loss_dict['cl_align'] + \
                    loss_dict['cl_intra'] + loss_dict['cl_disen']
            loss1.backward(retain_graph=True)

            # estimator
            est_loss = loss_dict['est']
            wall_loss = loss_dict['wall']
            if self.meters.check():
                set_requires_grad(self.model.encoder_r, requires_grad=False)
                est_loss.backward(retain_graph=True)
                set_requires_grad(self.model.encoder_r, requires_grad=True)
            else:
                set_requires_grad(self.model.encoder_ir, requires_grad=False)
                est_loss.backward(retain_graph=True)
                set_requires_grad(self.model.encoder_ir, requires_grad=True)

            wall_loss.backward(retain_graph=True)

            # discriminator
            loss_disc = loss_dict['disc']
            loss_disc.backward(retain_graph=True)

            self.optimizers_main.step()

            loss = loss1 + est_loss + wall_loss + loss_disc

            tr_loss += loss.item()
            nb_tr_steps += 1

        train_loss = tr_loss / nb_tr_steps
        # show every aspect of loss
        # loss_info = {k: round(v.item(), 4) for k, v in loss_dict.items()}
        # self.logger.info(f"{loss_info}")

        # Train Estimator
        self.train_estimator()
        # Train discriminator
        self.train_discriminator()

        return train_loss

    def test_one_epoch(self, mode='valid', return_embed_np=False):
        self.model.eval()

        preds = []
        labels = []
        total_embedding_list = []
        loss = 0
        with torch.no_grad():
            for step, batch in enumerate(self.dataloader[mode]):
                # load data
                input_ids, visual, acoustic, input_mask, segment_ids, label_ids = self.parse_batch(batch)

                results = self.model(input_ids, visual, acoustic, input_mask, segment_ids, label_ids)
                loss_dict = self.get_total_loss(results, label_ids)
                loss += sum(loss_dict.values()).item()

                task_output = results['task_output']
                logits = task_output.detach().cpu().numpy()
                label_ids = label_ids.detach().cpu().numpy()

                logits = np.squeeze(logits).tolist()
                label_ids = np.squeeze(label_ids).tolist()

                preds.extend(logits)
                labels.extend(label_ids)

                if return_embed_np:
                    embedding = torch.cat([
                        results['feature_r']['l'], results['feature_r']['a'], results['feature_r']['v'],
                        results['feature_ir']['l'], results['feature_ir']['a'], results['feature_ir']['v'],
                        results['fusion_f']
                    ], dim=-1)
                    total_embedding_list.append(embedding)

            preds = np.array(preds)
            labels = np.array(labels)
            test_loss = loss / (step + 1)

        if return_embed_np:
            return {'preds': preds, 'labels': labels, 'loss': test_loss,
                    'embedding': torch.cat(total_embedding_list, dim=0).detach().cpu().numpy()}
        else:
            return {'preds': preds, 'labels': labels, 'loss': test_loss}

    def process(self, log_path, result_path, return_embed_np=False):
        self.logger.info(
            "======================================== Program Start ========================================")
        self.logger.info("Running with args:")
        self.logger.info(self.config)
        self.logger.info(f"{'-' * 30} Running with seed {self.config.seed} {'-' * 30}")

        # num_param = self.count_parameters()
        # self.logger.info(f'The number of parameters: {num_param}')
        #
        # flops = self.compute_flops()
        # self.logger.info(f'The FLOPs: {flops}')

        best_loss = 100
        bset_eval_results = {}
        for epoch_i in range(int(self.config.n_epochs)):
            train_loss = self.train_one_epoch()
            valid_results = self.test_one_epoch(mode='valid')
            test_results = self.test_one_epoch(mode='test')

            metric_results = self.metrics(test_results['preds'], test_results['labels'])

            if return_embed_np:
                if epoch_i == 0:
                    np.save(os.path.join(log_path, 'embedding_before.npy'), test_results['embedding'])
                if (epoch_i + 1) == self.config.n_epochs:
                    np.save(os.path.join(log_path, 'embedding_last.npy'), test_results['embedding'])

            if valid_results['loss'] < best_loss:
                best_loss = valid_results['loss']
                bset_eval_results = {k: v for k, v in metric_results.items()}

                if return_embed_np:
                    np.save(os.path.join(log_path, 'embedding_best.npy'), test_results['embedding'])

                torch.save(self.model.state_dict(), os.path.join(log_path, f'best_model.pth'))

            self.logger.info(
                f"Epoch[{epoch_i + 1}/{self.config.n_epochs}] >> {dict_to_str(metric_results)} Train loss: {round(train_loss, 4)}  Valid loss: {round(valid_results['loss'], 4)}")

            if (epoch_i + 1) == self.config.n_epochs:
                self.logger.info(f"Best Results >> {dict_to_str(bset_eval_results)}")

                criterions = list(bset_eval_results.keys())
                arg_keys = list(self.config.keys())

                delete_param = ['TEXT_DIM', 'ACOUSTIC_DIM', 'VISUAL_DIM', 'max_seq_length', 'irrelevant_std',
                                'relevant_std', 'emb_radius', 'disc_limit_acc', 'est_ir_std', 'est_r_std',
                                'est_style_optimize', 'log', 'result', 'csv_file', 'data_path', 'pretrained']
                log_param = [item for item in arg_keys if item not in delete_param]

                # save result to csv
                csv_file = os.path.join(result_path, self.config.csv_file)
                if os.path.isfile(csv_file):
                    df = pd.read_csv(csv_file)
                else:
                    df = pd.DataFrame(columns=["Filename"] + criterions + log_param)

                filename = os.path.basename(log_path.rstrip('/'))

                res = [filename]
                for c in criterions:
                    values = bset_eval_results[c]
                    if c != 'MAE' and c != 'Corr':
                        res.append(round(values * 100, 2))
                    else:
                        res.append(values)

                for p in log_param:
                    params = self.config[p]
                    res.append(params)

                df.loc[len(df)] = res
                df.to_csv(csv_file, index=None)
                self.logger.info(f"Results saved to {csv_file}.")

    def get_total_loss(self, results, label_ids):
        feature_map = results['feature_map']
        feature_r = results['feature_r']
        feature_ir = results['feature_ir']
        task_output = results['task_output']
        decode_output = results['decode_output']
        recon_output = results['recon_output']
        est_output = results['est_output']
        disc_output = results['disc_output']

        # main loss
        loss_task = self.config.lambda_task * self.loss_task(task_output.view(-1), label_ids.view(-1))
        loss_dec = self.config.lambda_dec * (self.loss_task(decode_output['l'].view(-1), label_ids.view(-1)) +
                                             self.loss_task(decode_output['a'].view(-1), label_ids.view(-1)) +
                                             self.loss_task(decode_output['v'].view(-1), label_ids.view(-1)))
        loss_rec = self.config.lambda_rec * (self.loss_recon(recon_output['l'], feature_map['l']) +
                                             self.loss_recon(recon_output['a'], feature_map['a']) +
                                             self.loss_recon(recon_output['v'], feature_map['v']))
        # contrast learning loss
        loss_cl_align = self.config.lambda_cl_align * (
                self.loss_cl(mode='align', anchor=feature_r['l'], sample=feature_r['a']) +
                self.loss_cl(mode='align', anchor=feature_r['l'], sample=feature_r['v']))
        loss_cl_intra = self.config.lambda_cl_intra * self.loss_cl(mode='intra', data=feature_r['l'], label=label_ids)
        loss_cl_dis = self.config.lambda_cl_disentangle * (
                self.loss_cl(mode='disentangle', relevant=feature_r['l'], irrelevant=feature_ir['l']) +
                self.loss_cl(mode='disentangle', relevant=feature_r['a'], irrelevant=feature_ir['a']) +
                self.loss_cl(mode='disentangle', relevant=feature_r['v'], irrelevant=feature_ir['v']))
        # estimator
        loss_est_l = self.loss_est(output=est_output['l'], emb=(feature_ir['l'], feature_r['l']), mode='main')
        loss_est_a = self.loss_est(output=est_output['a'], emb=(feature_ir['a'], feature_r['a']), mode='main')
        loss_est_v = self.loss_est(output=est_output['v'], emb=(feature_ir['v'], feature_r['v']), mode='main')

        est_loss = self.config.lambda_est * (loss_est_l['loss_est'] + loss_est_a['loss_est'] + loss_est_v['loss_est'])
        wall_loss = self.config.lambda_wall * (
                loss_est_l['loss_wall'] + loss_est_a['loss_wall'] + loss_est_v['loss_wall'])
        # discriminator
        loss_disc = self.config.lambda_disc * (self.loss_disc(disc_output['l'], True)['loss'] +
                                               self.loss_disc(disc_output['a'], True)['loss'] +
                                               self.loss_disc(disc_output['v'], True)['loss'])

        return {
            'task': loss_task,
            'decode': loss_dec,
            'est': est_loss,
            'wall': wall_loss,
            'disc': loss_disc,
            'reconstruct': loss_rec,
            'cl_align': loss_cl_align,
            'cl_intra': loss_cl_intra,
            'cl_disen': loss_cl_dis
        }

    def parse_batch(self, batch):
        if self.dataset_name == 'sims':
            visual = batch['vision'].to(device)
            acoustic = batch['audio'].to(device)
            text = batch['text'].to(device)

            input_ids = text[:, 0, :].long().to(device)
            input_mask = text[:, 1, :].float().to(device)
            segment_ids = text[:, 2, :].long().to(device)

            label_ids = batch['labels']['M'].to(device)
            label_ids = label_ids.view(-1, 1, 1)
        else:
            batch = tuple(t.cuda() for t in batch)
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            # input_ids/input_mask/segment_ids:[bs, 50]; visual:[bs, 50, 47]; acoustic:[bs, 50, 74];label_ids.shape:[bs, 1, 1]
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

        return input_ids, visual, acoustic, input_mask, segment_ids, label_ids

    def train_estimator(self):
        self.model.train()

        for step, batch in enumerate(self.dataloader['train']):
            # load data
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = self.parse_batch(batch)

            # set optimizer
            set_requires_grad([self.model.extractor, self.model.encoder_r, self.model.encoder_ir,
                               self.model.fusion, self.model.decoder, self.model.reconstructor,
                               self.model.classifier], requires_grad=False)
            set_requires_grad(self.model.estimator, requires_grad=True)
            self.optimizers_est.zero_grad()

            # extract feature
            results = self.model(input_ids, visual, acoustic, input_mask, segment_ids, label_ids)
            feature_map = results['feature_map']
            feature_detach = dict2detach(feature_map)

            # Get embedding
            feature_r = self.model.encoder_r(feature_detach)
            feature_ir = self.model.encoder_ir(feature_detach)
            feature_r_detach = dict2detach(feature_r)
            feature_ir_detach = dict2detach(feature_ir)

            # estimator
            est_output_real = self.model.estimator(feature_r_detach, feature_ir_detach, mode='perm')
            est_output_fake = self.model.estimator(feature_r_detach, feature_ir_detach, mode='orig')

            crit_est_l = self.loss_est(output_fake=est_output_fake['l'], output_real=est_output_real['l'],
                                       mode='est')
            crit_est_a = self.loss_est(output_fake=est_output_fake['a'], output_real=est_output_real['a'],
                                       mode='est')
            crit_est_v = self.loss_est(output_fake=est_output_fake['v'], output_real=est_output_real['v'],
                                       mode='est')

            crit_est = (crit_est_l['loss_real'] + crit_est_a['loss_real'] + crit_est_v['loss_real']) + \
                       (crit_est_l['loss_fake'] + crit_est_a['loss_fake'] + crit_est_v['loss_fake']) + \
                       self.config.lambda_est_zc * (
                               crit_est_l['loss_zc'] + crit_est_a['loss_zc'] + crit_est_v['loss_zc'])
            crit_est.backward()
            self.optimizers_est.step()

    def train_discriminator(self):
        self.model.train()

        for step, batch in enumerate(self.dataloader['train']):
            # load data
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = self.parse_batch(batch)

            # set optimizer
            set_requires_grad([self.model.extractor, self.model.encoder_r, self.model.encoder_ir,
                               self.model.fusion, self.model.decoder, self.model.reconstructor,
                               self.model.classifier], requires_grad=False)
            set_requires_grad(self.model.disriminator, requires_grad=True)
            self.optimizers_disc.zero_grad()

            # extract feature
            results = self.model(input_ids, visual, acoustic, input_mask, segment_ids, label_ids)
            feature_map = results['feature_map']
            feature_detach = dict2detach(feature_map)

            # Get embedding
            feature_ir = self.model.encoder_ir(feature_detach)
            feature_ir_detach = dict2detach(feature_ir)
            recon_output = self.model.reconstructor(feature_ir_detach, label_ids)
            recon_detach = dict2detach(recon_output)

            disc_output_real = self.model.disriminator(feature_detach)
            disc_output_fake = self.model.disriminator(recon_detach)

            crit_disc_real_l = self.loss_disc(disc_output_real['l'], True)
            crit_disc_real_a = self.loss_disc(disc_output_real['a'], True)
            crit_disc_real_v = self.loss_disc(disc_output_real['v'], True)
            crit_disc_fake_l = self.loss_disc(disc_output_fake['l'], False)
            crit_disc_fake_a = self.loss_disc(disc_output_fake['a'], False)
            crit_disc_fake_v = self.loss_disc(disc_output_fake['v'], False)

            pred_list = [
                crit_disc_real_l['pred'] == 1,
                crit_disc_real_a['pred'] == 1,
                crit_disc_real_v['pred'] == 1,
                crit_disc_fake_l['pred'] == 0,
                crit_disc_fake_a['pred'] == 0,
                crit_disc_fake_v['pred'] == 0
            ]
            disc_acc = torch.cat(pred_list, dim=0).sum().item() / (input_ids.size(0) * 6)
            if disc_acc < self.config.disc_limit_acc:
                crit_disc = (crit_disc_real_l['loss'] + crit_disc_real_a['loss'] + crit_disc_real_v['loss']) + \
                            (crit_disc_fake_l['loss'] + crit_disc_fake_a['loss'] + crit_disc_fake_v['loss'])
                crit_disc.backward()
                self.optimizers_disc.step()

    def count_parameters(self):
        res = 0
        for p in self.model.parameters():
            if p.requires_grad:
                res += p.numel()
        return res

    def compute_flops(self):
        # The number and size of parameters need to be adjusted according to the actual model receiving parameters.
        self.model.eval()

        input_ids = torch.randint(low=0, high=self.config.TEXT_DIM, size=(1, 50), dtype=torch.long).cuda()
        input_mask = torch.ones((1, 50), dtype=torch.long).cuda()
        segment_ids = torch.ones((1, 50), dtype=torch.long).cuda()

        visual = torch.randn((1, 50, self.config.VISUAL_DIM)).cuda()
        acoustic = torch.randn((1, 50, self.config.ACOUSTIC_DIM)).cuda()
        label_ids = torch.randn((1, 1, 1)).cuda()

        input_tuple = (input_ids, visual, acoustic, input_mask, segment_ids, label_ids)
        flops = profile_macs(self.model, input_tuple)

        return flops


if __name__ == '__main__':
    start = datetime.now()
    # set random seed
    set_random_seed(args.seed)

    log_name = "{}_{}_{}_{}".format(
        args.dataset_name,
        args.seed,
        start.strftime("%Y-%m-%d-%H_%M_%S"),
        str(random.randint(0, 10000))
    )

    # check dir
    save_path = os.path.join(args.log, log_name)
    result_path = args.result
    mkdirs(args.log, save_path, result_path)

    # logs setting
    log = set_logger(save_path, log_name)

    configs = {}
    configs.update(vars(args))
    configs.update(dataset_param[args.dataset_name])

    with open(os.path.join(save_path, 'args.json'), 'w') as f:
        json.dump(configs, f, indent=4)

    configs = edict(configs)

    # load data
    dataloader = set_up_data_loader(configs)

    # model training
    Train(dataloader, configs, log).process(save_path, result_path)
