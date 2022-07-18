from . import BaseActor
import torch
import numpy as np 
from scipy.linalg import block_diag
import pdb



class CIDiMPActor(BaseActor):
    """Actor for training the CIA network."""
    def __init__(self, net, objective, loss_weight=None, upper_cls_map=None, q_only=False):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'iou': 1.0, 'test_clf': 1.0, 'upper_clf': 1.0}
        self.loss_weight = loss_weight
        self.q_only = q_only 
        self.upper_cls_map = upper_cls_map 

    def _upper_cls_map(self, cls):
        if cls in self.upper_cls_map:
            return self.upper_cls_map[cls]
        else:
            return -1

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        num_imgs_train, batch_size = data['train_images'].shape[0:2]
        num_imgs_test = data['test_images'].shape[0]

        all_imgs = torch.cat([data['train_images'], data['test_images']], dim=0)
        all_bb = torch.cat([data['train_anno'], data['test_anno']], dim=0)

        # Shuffle for making use of BN
        idx_shuffle = torch.randperm(batch_size).cuda()
        shuffled_imgs = all_imgs[:, idx_shuffle, :, :, :]
        shuffled_bb = all_bb[:, idx_shuffle, :]
        idx_unshuffle = torch.argsort(idx_shuffle)

        upper_cls_pred, target_scores, iou_pred,\
        q_vec, shuffled_k_vec = self.net(train_imgs=data['train_images'],
                                         test_imgs=data['test_images'],
                                         train_bb=data['train_anno'],
                                         test_bb=data['test_anno'],
                                         test_proposals=data['test_proposals'],
                                         shuffled_imgs=shuffled_imgs,
                                         shuffled_bb=shuffled_bb)
                                        #  seq_name=data['seq_name'])  # for cls visualization

        q_vec = q_vec.permute(1, 0, 2).reshape(-1, q_vec.shape[-1])
        # Undo shuffle
        shuffled_k_vec = shuffled_k_vec.permute(1, 0, 2)
        if self.q_only:
            k_vec = shuffled_k_vec.reshape(-1, shuffled_k_vec.shape[-1])
        else:
            k_vec = shuffled_k_vec[idx_unshuffle, :, :].reshape(-1, shuffled_k_vec.shape[-1])

        # MOCO similarity matrix
        moco_sim_matrix = torch.einsum('nc,mc->nm', [q_vec, k_vec])

        # Contrastive label
        ones_label = np.ones((num_imgs_train+num_imgs_test, num_imgs_train+num_imgs_test), dtype=np.float32)
        ones_label_list = [ones_label] * batch_size 
        moco_label_matrix = torch.tensor(block_diag(*ones_label_list)).to(moco_sim_matrix.device)

        # Contrastive loss
        loss_moco = 0
        if 'contrastive' in self.loss_weight.keys():
            loss_moco = self.loss_weight['contrastive'] * self.objective['contrastive'](moco_sim_matrix, moco_label_matrix)

        # Upper classification loss
        loss_upper_clf = 0
        if 'upper_clf' in self.loss_weight.keys():
            upper_cls_gt = []
            for i in range(len(data['test_class'])):
                upper_cls_gt.append(self._upper_cls_map(data['test_class'][i]))
            upper_cls_gt = upper_cls_gt * 6
            upper_cls_pred = upper_cls_pred.reshape(-1, upper_cls_pred.shape[-1])
            loss_upper_clf = self.loss_weight['upper_clf'] * self.objective['upper_clf'](upper_cls_pred, upper_cls_gt)

        # Classification losses for the different optimization iterations
        clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

        # Loss of the final filter
        clf_loss_test = clf_losses_test[-1]
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

        # Compute loss for ATOM IoUNet
        loss_iou = self.loss_weight['iou'] * self.objective['iou'](iou_pred, data['proposal_iou'])
        # loss_upper_clf = loss_iou 

        # Loss for the initial filter iteration
        loss_test_init_clf = 0
        if 'test_init_clf' in self.loss_weight.keys():
            loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

        # Loss for the intermediate filter iterations
        loss_test_iter_clf = 0
        if 'test_iter_clf' in self.loss_weight.keys():
            test_iter_weights = self.loss_weight['test_iter_clf']
            if isinstance(test_iter_weights, list):
                loss_test_iter_clf = sum([a*b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
            else:
                loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # Total loss
        loss = loss_iou + loss_target_classifier + \
                loss_test_init_clf + loss_test_iter_clf + \
                loss_upper_clf + loss_moco 

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss_iou.item(),
                 'Loss/target_clf': loss_target_classifier.item()}
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        stats['ClfTrain/test_loss'] = clf_loss_test.item()
        if len(clf_losses_test) > 0:
            stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
            if len(clf_losses_test) > 2:
                stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)
        if 'upper_clf' in self.loss_weight.keys():
            stats['Loss/upper_clf'] = loss_upper_clf.item()
        if 'contrastive' in self.loss_weight.keys():
            stats['Loss/contrastive'] = loss_moco.item()

        return loss, stats


class CIKLDiMPActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None, upper_cls_map=None, q_only=False):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0, 'test_clf': 1.0, 'upper_clf': 1.0}
        self.loss_weight = loss_weight
        self.q_only = q_only 
        self.upper_cls_map = upper_cls_map 

    def _upper_cls_map(self, cls):
        if cls in self.upper_cls_map:
            return self.upper_cls_map[cls]
        else:
            return -1

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        num_imgs_train, batch_size = data['train_images'].shape[0:2]
        num_imgs_test = data['test_images'].shape[0]

        all_imgs = torch.cat([data['train_images'], data['test_images']], dim=0)
        all_bb = torch.cat([data['train_anno'], data['test_anno']], dim=0)

        # Shuffle for making use of BN
        idx_shuffle = torch.randperm(batch_size).cuda()
        shuffled_imgs = all_imgs[:, idx_shuffle, :, :]
        shuffled_bb = all_bb[:, idx_shuffle, :]
        idx_unshuffle = torch.argsort(idx_shuffle)

        upper_cls_pred, target_scores, bb_scores,\
        q_vec, shuffled_k_vec = self.net(train_imgs=data['train_images'],
                                         test_imgs=data['test_images'],
                                         train_bb=data['train_anno'],
                                         test_bb=data['test_anno'],
                                         test_proposals=data['test_proposals'],
                                         shuffled_imgs=shuffled_imgs,
                                         shuffled_bb=shuffled_bb)

        q_vec = q_vec.permute(1, 0, 2).reshape(-1, q_vec.shape[-1])
        # Undo shuffle
        shuffled_k_vec = shuffled_k_vec.permute(1, 0, 2)
        if self.q_only:
            k_vec = shuffled_k_vec.reshape(-1, shuffled_k_vec.shape[-1])
        else:
            k_vec = shuffled_k_vec[idx_unshuffle, :, :].reshape(-1, shuffled_k_vec.shape[-1])

        # MOCO similarity matrix
        moco_sim_matrix = torch.einsum('nc,mc->nm', [q_vec, k_vec])

        # Contrastive label
        ones_label = np.ones((num_imgs_train+num_imgs_test, num_imgs_train+num_imgs_test), dtype=np.float32)
        ones_label_list = [ones_label] * batch_size
        moco_label_matrix = torch.tensor(block_diag(*ones_label_list)).to(moco_sim_matrix.device)

        # Contrastive loss
        loss_moco = 0
        if 'contrastive' in self.loss_weight.keys():
            loss_moco = self.loss_weight['contrastive'] * self.objective['contrastive'](moco_sim_matrix, moco_label_matrix)

        # Upper classification loss
        loss_upper_clf = 0
        if 'upper_clf' in self.loss_weight.keys():
            upper_cls_gt = []
            for i in range(len(data['test_class'])):
                upper_cls_gt.append(self._upper_cls_map(data['test_class'][i]))
            upper_cls_gt = upper_cls_gt * 6
            upper_cls_pred = upper_cls_pred.reshape(-1, upper_cls_pred.shape[-1])
            loss_upper_clf = self.loss_weight['upper_clf'] * self.objective['upper_clf'](upper_cls_pred, upper_cls_gt)

        # Reshape bb reg variables
        is_valid = data['test_anno'][:, :, 0] < 99999.0
        bb_scores = bb_scores[is_valid, :]
        proposal_density = data['proposal_density'][is_valid, :]
        gt_density = data['gt_density'][is_valid, :]

        # Compute loss
        bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce

        # If standard DiMP classifier is used
        loss_target_classifier = 0
        loss_test_init_clf = 0
        loss_test_iter_clf = 0
        if 'test_clf' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

            # Loss of the final filter
            clf_loss_test = clf_losses_test[-1]
            loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

            # Loss for the initial filter iteration
            if 'test_init_clf' in self.loss_weight.keys():
                loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

            # Loss for the intermediate filter iterations
            if 'test_iter_clf' in self.loss_weight.keys():
                test_iter_weights = self.loss_weight['test_iter_clf']
                if isinstance(test_iter_weights, list):
                    loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
                else:
                    loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # If PrDiMP classifier is used
        loss_clf_ce = 0
        loss_clf_ce_init = 0
        loss_clf_ce_iter = 0
        if 'clf_ce' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_ce_losses = [self.objective['clf_ce'](s, data['test_label_density'], grid_dim=(-2,-1)) for s in target_scores]

            # Loss of the final filter
            clf_ce = clf_ce_losses[-1]
            loss_clf_ce = self.loss_weight['clf_ce'] * clf_ce

            # Loss for the initial filter iteration
            if 'clf_ce_init' in self.loss_weight.keys():
                loss_clf_ce_init = self.loss_weight['clf_ce_init'] * clf_ce_losses[0]

            # Loss for the intermediate filter iterations
            if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
                test_iter_weights = self.loss_weight['clf_ce_iter']
                if isinstance(test_iter_weights, list):
                    loss_clf_ce_iter = sum([a * b for a, b in zip(test_iter_weights, clf_ce_losses[1:-1])])
                else:
                    loss_clf_ce_iter = (test_iter_weights / (len(clf_ce_losses) - 2)) * sum(clf_ce_losses[1:-1])

        # Total loss
        loss = loss_bb_ce + loss_clf_ce + loss_clf_ce_init + loss_clf_ce_iter + \
                loss_target_classifier + loss_test_init_clf + loss_test_iter_clf + \
                loss_upper_clf + loss_moco 

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': bb_ce.item(),
                 'Loss/loss_bb_ce': loss_bb_ce.item()}
        if 'test_clf' in self.loss_weight.keys():
            stats['Loss/target_clf'] = loss_target_classifier.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        if 'clf_ce' in self.loss_weight.keys():
            stats['Loss/clf_ce'] = loss_clf_ce.item()
        if 'clf_ce_init' in self.loss_weight.keys():
            stats['Loss/clf_ce_init'] = loss_clf_ce_init.item()
        if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
            stats['Loss/clf_ce_iter'] = loss_clf_ce_iter.item()

        if 'test_clf' in self.loss_weight.keys():
            stats['ClfTrain/test_loss'] = clf_loss_test.item()
            if len(clf_losses_test) > 0:
                stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
                if len(clf_losses_test) > 2:
                    stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        if 'clf_ce' in self.loss_weight.keys():
            stats['ClfTrain/clf_ce'] = clf_ce.item()
            if len(clf_ce_losses) > 0:
                stats['ClfTrain/clf_ce_init'] = clf_ce_losses[0].item()
                if len(clf_ce_losses) > 2:
                    stats['ClfTrain/clf_ce_iter'] = sum(clf_ce_losses[1:-1]).item() / (len(clf_ce_losses) - 2)

        if 'upper_clf' in self.loss_weight.keys():
            stats['Loss/upper_clf'] = loss_upper_clf.item()
        if 'contrastice' in self.loss_weight.keys():
            stats['Loss/contrastive'] = loss_moco.item()

        return loss, stats


class DiMPActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'iou': 1.0, 'test_clf': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, iou_pred = self.net(train_imgs=data['train_images'],
                                           test_imgs=data['test_images'],
                                           train_bb=data['train_anno'],
                                           test_proposals=data['test_proposals'])

        # Classification losses for the different optimization iterations
        clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

        # Loss of the final filter
        clf_loss_test = clf_losses_test[-1]
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

        # Compute loss for ATOM IoUNet
        loss_iou = self.loss_weight['iou'] * self.objective['iou'](iou_pred, data['proposal_iou'])

        # Loss for the initial filter iteration
        loss_test_init_clf = 0
        if 'test_init_clf' in self.loss_weight.keys():
            loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

        # Loss for the intermediate filter iterations
        loss_test_iter_clf = 0
        if 'test_iter_clf' in self.loss_weight.keys():
            test_iter_weights = self.loss_weight['test_iter_clf']
            if isinstance(test_iter_weights, list):
                loss_test_iter_clf = sum([a*b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
            else:
                loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # Total loss
        loss = loss_iou + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss_iou.item(),
                 'Loss/target_clf': loss_target_classifier.item()}
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        stats['ClfTrain/test_loss'] = clf_loss_test.item()
        if len(clf_losses_test) > 0:
            stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
            if len(clf_losses_test) > 2:
                stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        return loss, stats


class KLDiMPActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        target_scores, bb_scores = self.net(train_imgs=data['train_images'],
                                            test_imgs=data['test_images'],
                                            train_bb=data['train_anno'],
                                            test_proposals=data['test_proposals'])

        # Reshape bb reg variables
        is_valid = data['test_anno'][:, :, 0] < 99999.0
        bb_scores = bb_scores[is_valid, :]
        proposal_density = data['proposal_density'][is_valid, :]
        gt_density = data['gt_density'][is_valid, :]

        # Compute loss
        bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce

        # If standard DiMP classifier is used
        loss_target_classifier = 0
        loss_test_init_clf = 0
        loss_test_iter_clf = 0
        if 'test_clf' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

            # Loss of the final filter
            clf_loss_test = clf_losses_test[-1]
            loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

            # Loss for the initial filter iteration
            if 'test_init_clf' in self.loss_weight.keys():
                loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

            # Loss for the intermediate filter iterations
            if 'test_iter_clf' in self.loss_weight.keys():
                test_iter_weights = self.loss_weight['test_iter_clf']
                if isinstance(test_iter_weights, list):
                    loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
                else:
                    loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # If PrDiMP classifier is used
        loss_clf_ce = 0
        loss_clf_ce_init = 0
        loss_clf_ce_iter = 0
        if 'clf_ce' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_ce_losses = [self.objective['clf_ce'](s, data['test_label_density'], grid_dim=(-2,-1)) for s in target_scores]

            # Loss of the final filter
            clf_ce = clf_ce_losses[-1]
            loss_clf_ce = self.loss_weight['clf_ce'] * clf_ce

            # Loss for the initial filter iteration
            if 'clf_ce_init' in self.loss_weight.keys():
                loss_clf_ce_init = self.loss_weight['clf_ce_init'] * clf_ce_losses[0]

            # Loss for the intermediate filter iterations
            if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
                test_iter_weights = self.loss_weight['clf_ce_iter']
                if isinstance(test_iter_weights, list):
                    loss_clf_ce_iter = sum([a * b for a, b in zip(test_iter_weights, clf_ce_losses[1:-1])])
                else:
                    loss_clf_ce_iter = (test_iter_weights / (len(clf_ce_losses) - 2)) * sum(clf_ce_losses[1:-1])

        # Total loss
        loss = loss_bb_ce + loss_clf_ce + loss_clf_ce_init + loss_clf_ce_iter + \
                            loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': bb_ce.item(),
                 'Loss/loss_bb_ce': loss_bb_ce.item()}
        if 'test_clf' in self.loss_weight.keys():
            stats['Loss/target_clf'] = loss_target_classifier.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()
        if 'clf_ce' in self.loss_weight.keys():
            stats['Loss/clf_ce'] = loss_clf_ce.item()
        if 'clf_ce_init' in self.loss_weight.keys():
            stats['Loss/clf_ce_init'] = loss_clf_ce_init.item()
        if 'clf_ce_iter' in self.loss_weight.keys() and len(clf_ce_losses) > 2:
            stats['Loss/clf_ce_iter'] = loss_clf_ce_iter.item()

        if 'test_clf' in self.loss_weight.keys():
            stats['ClfTrain/test_loss'] = clf_loss_test.item()
            if len(clf_losses_test) > 0:
                stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
                if len(clf_losses_test) > 2:
                    stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        if 'clf_ce' in self.loss_weight.keys():
            stats['ClfTrain/clf_ce'] = clf_ce.item()
            if len(clf_ce_losses) > 0:
                stats['ClfTrain/clf_ce_init'] = clf_ce_losses[0].item()
                if len(clf_ce_losses) > 2:
                    stats['ClfTrain/clf_ce_iter'] = sum(clf_ce_losses[1:-1]).item() / (len(clf_ce_losses) - 2)

        return loss, stats
