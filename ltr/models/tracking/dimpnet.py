import math
import torch
import torch.nn as nn
from collections import OrderedDict
from ltr.models.meta import steepestdescent
import ltr.models.upper_classifier.upper_classifier as upper_clf 
import ltr.models.target_classifier.linear_filter as target_clf
import ltr.models.target_classifier.features as clf_features
import ltr.models.target_classifier.initializer as clf_initializer
import ltr.models.target_classifier.optimizer as clf_optimizer
import ltr.models.bbreg as bbmodels
import ltr.models.backbone as backbones
from ltr import model_constructor
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from ltr.models.utils import bb2roi 
import os 
import cv2 
import numpy as np

import pdb


class CIDiMPnet(nn.Module):
    """The DiMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        classifier:  Target classification module.
        bb_regressor:  Bounding box regression module.
        classification_layer:  Name of the backbone feature layer to use for classification.
        bb_regressor_layer:  Names of the backbone layers to use for bounding box regression."""

    def __init__(self, feature_extractor, upper_classifier, classifier, bb_regressor, upper_layer, classification_layer, bb_regressor_layer, refine_flag=True):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.upper_classifier = upper_classifier
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.upper_layer = upper_layer
        if 'ins' in classification_layer:
            classification_layer = classification_layer[0:-4]
            self.clf_ins = True 
        else:
            self.clf_ins = False 
        self.classification_layer = [classification_layer] if isinstance(classification_layer, str) else classification_layer
        self.bb_regressor_layer = bb_regressor_layer
        self.output_layers = sorted(list(set(self.upper_layer + self.classification_layer + self.bb_regressor_layer)))
        self.refine_flag = refine_flag 


    def forward(self, train_imgs, test_imgs, train_bb, test_bb, test_proposals, shuffled_imgs, shuffled_bb, *args, **kwargs):
        """Runs the DiMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            test_proposals:  Proposal boxes to use for the IoUNet (bb_regressor) module.
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            iou_pred:  Predicted IoU scores for the test_proposals."""

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        # Extract backbone features
        num_imgs_train = train_imgs.shape[0]
        num_imgs_test = test_imgs.shape[0]
        batch_size = train_imgs.shape[1]

        all_imgs = torch.cat([train_imgs, test_imgs], dim=0)
        all_bb = torch.cat([train_bb, test_bb], dim=0)

        all_feat, q_vec, shuffled_k_vec = self.extract_backbone_features(all_imgs, all_bb, shuffled_imgs, shuffled_bb)
        q_vec = q_vec.reshape(batch_size, num_imgs_train+num_imgs_test, -1).permute(1, 0, 2)
        shuffled_k_vec = shuffled_k_vec.reshape(batch_size, num_imgs_train+num_imgs_test, -1).permute(1, 0, 2) 

        train_feat = OrderedDict()
        test_feat = OrderedDict()
        for key in all_feat.keys():
            all_feat_now = all_feat[key]
            all_feat_now = all_feat_now.reshape(batch_size, num_imgs_train+num_imgs_test, *all_feat_now.shape[-3:]).permute(1, 0, 2, 3, 4)
            train_feat[key] = all_feat_now[0:num_imgs_train].reshape(-1, *all_feat_now.shape[-3:])
            test_feat[key] = all_feat_now[num_imgs_train:].reshape(-1, *all_feat_now.shape[-3:])

        # Upper classification features
        train_feat_upper = self.get_backbone_upper_feat(train_feat)
        test_feat_upper = self.get_backbone_upper_feat(test_feat)

        # Run upper classifier module
        bb1 = train_bb.reshape(-1, 4)
        bb2 = test_bb.reshape(-1, 4)
        bb = torch.cat([bb1, bb2], dim=0)
        feat_upper = torch.cat([train_feat_upper, test_feat_upper], dim=0)
        upper_cls_pred, upper_cls_center = self.upper_classifier(feat_upper, bb)
        upper_cls_pred = upper_cls_pred.reshape(-1, train_imgs.shape[1], upper_cls_pred.shape[-1])

        # Classification features
        self.clf_ins = False # for ablation study 
        train_feat_clf = self.get_backbone_clf_feat(train_feat)
        test_feat_clf = self.get_backbone_clf_feat(test_feat)

        # Run refiner module
        self.refine_flag = False # for ablation study
        if self.refine_flag:
            feat_trans = torch.cat([train_feat_clf, test_feat_clf], dim=0)
            out_feat_clf = self.upper_classifier.semantic_trans(feat_trans)
            train_feat_clf = out_feat_clf[0:train_feat_clf.shape[0]]
            test_feat_clf = out_feat_clf[train_feat_clf.shape[0]:]

        # Run classifier module
        target_scores = self.classifier(train_feat_clf, test_feat_clf, train_bb, *args, **kwargs)

        # Get bb_regressor features
        train_feat_iou = self.get_backbone_bbreg_feat(train_feat)
        test_feat_iou = self.get_backbone_bbreg_feat(test_feat)

        # Run the IoUNet module
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou, train_bb, test_proposals)

        return upper_cls_pred, target_scores, iou_pred, q_vec, shuffled_k_vec 

    def get_backbone_clf_feat(self, backbone_feat):
        if self.clf_ins:
            feat = OrderedDict({l+'_ins': backbone_feat[l+'_ins'] for l in self.classification_layer})
        else:
            feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            if self.clf_ins:
                return feat[self.classification_layer[0] + '_ins']
            else:
                return feat[self.classification_layer[0]]
        return feat

    def get_backbone_upper_feat(self, backbone_feat):
        return backbone_feat[self.upper_layer[0]]

    def get_backbone_bbreg_feat(self, backbone_feat):
        return [backbone_feat[l] for l in self.bb_regressor_layer]

    def extract_classification_feat(self, backbone_feat):
        return self.classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im_q, bb_q=None, im_k=None, bb_k=None, layers=None):
        if layers is None:
            layers = self.output_layers
        if im_k is None:
            return self.feature_extractor.extract_backbone_feature(im_q, layers)
        return self.feature_extractor(im_q, im_k, bb_q, bb_k, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer + ['classification']
        if 'classification' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.classification_layer if l != 'classification'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_classification_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})


@model_constructor
def cia18(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
          classification_layer='layer3_ins', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=1,
          clf_feat_norm=True, init_filter_norm=False, final_conv=True,
          out_feature_dim=256, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
          mask_init_factor=4.0, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
          score_act='relu', act_param=None, target_mask_act='sigmoid',
          detach_length=float('Inf'), frozen_backbone_layers=()):
    """ add instance modulator v2(learned alpha) to encode instance information. 
    """
    # Backbone
    backbone_net = backbones.moco_resnet18(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    clf_feature_extractor = clf_features.residual_basic_block(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # The upper classifier module
    upper_classifier = upper_clf.FCClassifier_v2(in_channel=512)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = CIDiMPnet(feature_extractor=backbone_net, upper_classifier=upper_classifier, classifier=classifier, bb_regressor=bb_regressor,
                  upper_layer=['layer4'], classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net


@model_constructor
def cia50(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
          classification_layer='layer3_ins', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=1,
          clf_feat_norm=True, init_filter_norm=False, final_conv=True,
          out_feature_dim=256, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
          mask_init_factor=4.0, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
          score_act='relu', act_param=None, target_mask_act='sigmoid',
          detach_length=float('Inf'), frozen_backbone_layers=()):
    """ add instance modulator v2(learned alpha) to encode instance information.
        upper cls loss + upper cls encoder + instance loss + instance encoder
    """
    # Backbone
    backbone_net = backbones.moco_resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if 'layer3' in classification_layer:
        feature_dim = 256
    elif 'layer4' in classification_layer:
        feature_dim = 512
    else:
        raise Exception
    clf_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                             num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # The upper classifier module
    upper_classifier = upper_clf.FCClassifier_v2(in_channel=2048, post_proj_channel=1024)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(4*128,4*256), pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = CIDiMPnet(feature_extractor=backbone_net, upper_classifier=upper_classifier, classifier=classifier, bb_regressor=bb_regressor,
                  upper_layer=['layer4'], classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net


class DiMPnet(nn.Module):
    """The DiMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        classifier:  Target classification module.
        bb_regressor:  Bounding box regression module.
        classification_layer:  Name of the backbone feature layer to use for classification.
        bb_regressor_layer:  Names of the backbone layers to use for bounding box regression."""

    def __init__(self, feature_extractor, classifier, bb_regressor, classification_layer, bb_regressor_layer):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.classification_layer = [classification_layer] if isinstance(classification_layer, str) else classification_layer
        self.bb_regressor_layer = bb_regressor_layer
        self.output_layers = sorted(list(set(self.classification_layer + self.bb_regressor_layer)))


    def forward(self, train_imgs, test_imgs, train_bb, test_proposals, *args, **kwargs):
        """Runs the DiMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            test_proposals:  Proposal boxes to use for the IoUNet (bb_regressor) module.
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            iou_pred:  Predicted IoU scores for the test_proposals."""

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))

        # Classification features
        train_feat_clf = self.get_backbone_clf_feat(train_feat)
        test_feat_clf = self.get_backbone_clf_feat(test_feat)

        # Run classifier module
        target_scores = self.classifier(train_feat_clf, test_feat_clf, train_bb, *args, **kwargs)

        # Get bb_regressor features
        train_feat_iou = self.get_backbone_bbreg_feat(train_feat)
        test_feat_iou = self.get_backbone_bbreg_feat(test_feat)

        # Run the IoUNet module
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou, train_bb, test_proposals)

        return target_scores, iou_pred

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]
        return feat

    def get_backbone_bbreg_feat(self, backbone_feat):
        return [backbone_feat[l] for l in self.bb_regressor_layer]

    def extract_classification_feat(self, backbone_feat):
        return self.classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = self.bb_regressor_layer + ['classification']
        if 'classification' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.classification_layer if l != 'classification'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_classification_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})



@model_constructor
def dimpnet18(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
              classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=1,
              clf_feat_norm=True, init_filter_norm=False, final_conv=True,
              out_feature_dim=256, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
              mask_init_factor=4.0, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
              score_act='relu', act_param=None, target_mask_act='sigmoid',
              detach_length=float('Inf'), frozen_backbone_layers=()):
    # Backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    clf_feature_extractor = clf_features.residual_basic_block(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net


@model_constructor
def dimpnet50(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
              classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
              clf_feat_norm=True, init_filter_norm=False, final_conv=True,
              out_feature_dim=512, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
              mask_init_factor=4.0, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
              score_act='relu', act_param=None, target_mask_act='sigmoid',
              detach_length=float('Inf'), frozen_backbone_layers=()):

    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if classification_layer == 'layer3':
        feature_dim = 256
    elif classification_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    clf_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                             num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(4*128,4*256), pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net



@model_constructor
def L2dimpnet18(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
              classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=1,
              clf_feat_norm=True, init_filter_norm=False, final_conv=True,
              out_feature_dim=256, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
              detach_length=float('Inf'), hinge_threshold=-999, gauss_sigma=1.0, alpha_eps=0):
    # Backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    clf_feature_extractor = clf_features.residual_basic_block(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPL2SteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step, hinge_threshold=hinge_threshold,
                                                    init_filter_reg=optim_init_reg, gauss_sigma=gauss_sigma,
                                                    detach_length=detach_length, alpha_eps=alpha_eps)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net


@model_constructor
def klcedimpnet18(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
                  classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=1,
                  clf_feat_norm=True, init_filter_norm=False, final_conv=True,
                  out_feature_dim=256, gauss_sigma=1.0,
                  iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
                  detach_length=float('Inf'), alpha_eps=0.0, train_feature_extractor=True,
                  init_uni_weight=None, optim_min_reg=1e-3, init_initializer='default', normalize_label=False,
                  label_shrink=0, softmax_reg=None, label_threshold=0, final_relu=False, init_pool_square=False,
                  frozen_backbone_layers=()):

    if not train_feature_extractor:
        frozen_backbone_layers = 'all'

    # Backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    clf_feature_extractor = clf_features.residual_basic_block(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim, final_relu=final_relu)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim, init_weights=init_initializer,
                                                          pool_square=init_pool_square)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.PrDiMPSteepestDescentNewton(num_iter=optim_iter, feat_stride=feat_stride,
                                                          init_step_length=optim_init_step,
                                                          init_filter_reg=optim_init_reg, gauss_sigma=gauss_sigma,
                                                          detach_length=detach_length, alpha_eps=alpha_eps,
                                                          init_uni_weight=init_uni_weight,
                                                          min_filter_reg=optim_min_reg, normalize_label=normalize_label,
                                                          label_shrink=label_shrink, softmax_reg=softmax_reg,
                                                          label_threshold=label_threshold)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net


@model_constructor
def klcedimpnet50(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
                  classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
                  clf_feat_norm=True, init_filter_norm=False, final_conv=True,
                  out_feature_dim=512, gauss_sigma=1.0,
                  iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
                  detach_length=float('Inf'), alpha_eps=0.0, train_feature_extractor=True,
                  init_uni_weight=None, optim_min_reg=1e-3, init_initializer='default', normalize_label=False,
                  label_shrink=0, softmax_reg=None, label_threshold=0, final_relu=False, frozen_backbone_layers=()):

    if not train_feature_extractor:
        frozen_backbone_layers = 'all'

    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    clf_feature_extractor = clf_features.residual_bottleneck(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim, final_relu=final_relu)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim, init_weights=init_initializer)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.PrDiMPSteepestDescentNewton(num_iter=optim_iter, feat_stride=feat_stride,
                                                          init_step_length=optim_init_step,
                                                          init_filter_reg=optim_init_reg, gauss_sigma=gauss_sigma,
                                                          detach_length=detach_length, alpha_eps=alpha_eps,
                                                          init_uni_weight=init_uni_weight,
                                                          min_filter_reg=optim_min_reg, normalize_label=normalize_label,
                                                          label_shrink=label_shrink, softmax_reg=softmax_reg,
                                                          label_threshold=label_threshold)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(4*128,4*256), pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net
