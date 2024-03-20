import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .segmentation import (dice_loss, sigmoid_focal_loss)
import copy
from termcolor import colored

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, model_name, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.losses_for_replay = {}
        self.losses_for_replay['loss_bbox'] = []
        self.losses_for_replay['loss_giou'] = []
        self.losses_for_replay['loss_labels'] = []
        self.model_name = model_name

        
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        if outputs['gt'] is not None:
            # class index와 gt 맞춰주기 위함
            target_classes_o = torch.tensor(
                [outputs['gt'].index(target)+1 for target in target_classes_o], dtype=torch.int64
            ).to(target_classes_o.device)
        
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce, sample_focalloss_in_batch = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2)
        loss_ce = loss_ce * src_logits.shape[1] # loss scolar value * query counts(300)

        if self.buffer_construct_loss is True :
            # query multiplication and num_labels normalization term
            sample_focalloss_in_batch = sample_focalloss_in_batch * src_logits.shape[1] # Query counts = 300
            sample_focalloss_in_batch = sample_focalloss_in_batch / torch.tensor([t['labels'].shape[0] for t in targets],
                                                                                device=targets[0]['labels'].device)
            self.losses_for_replay['loss_labels'] = [each_loss for each_loss in sample_focalloss_in_batch]
            
        losses = {'loss_ce': loss_ce}
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if self.buffer_construct_loss is True :
            for i in range(len(indices)):
                # each batch's idx value to new tuple function
                i_idx = (idx[0][idx[0] == i], idx[1][idx[0] == i])
                if indices[i][0].nelement() == 0:
                    self.losses_for_replay['loss_bbox'].append(torch.tensor(100., device=targets[0]['boxes'].device))
                    self.losses_for_replay['loss_giou'].append(torch.tensor(100., device=targets[0]['boxes'].device))
                    print(colored(f"high loss input to each loss in batch, becuase no target", "red", "on_yellow"))
                    continue
                
                src_boxes = outputs['pred_boxes'][i_idx]
                target_boxes = targets[i]['boxes']
                each_bbox_count = target_boxes.shape[0]
                
                loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
                # Save the un-reduced L1 losses for each image
                loss_bbox = loss_bbox.sum()
                loss_bbox /= each_bbox_count
                self.losses_for_replay['loss_bbox'].append(loss_bbox)

                
                loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_boxes),
                    box_ops.box_cxcywh_to_xyxy(target_boxes)))
                
                if indices[i][0].nelement() == 0 :
                    # If not calc each loss both bbox or giou, so then we put high loss to temporary var
                    self.losses_for_replay['loss_bbox'].append(torch.tensor(100., device=targets[0]['boxes'].device))
                    self.losses_for_replay['loss_giou'].append(torch.tensor(100., device=targets[0]['boxes'].device))
                    print(colored(f"high loss input to each loss in batch, because no giou", "red", "on_yellow"))
                    continue
                
                loss_giou = loss_giou.sum()
                loss_giou /= each_bbox_count
                # Save the un-reduced GIoU losses for each image
                self.losses_for_replay['loss_giou'].append(loss_giou)
            
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        if self.model_name == 'dn_detr':
            with torch.no_grad():
                losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
                losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]


        if self.model_name == 'deform_detr':
            # TODO use valid to mask invalid areas due to padding in loss
            target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
            target_masks = target_masks.to(src_masks)
            src_masks = src_masks[src_idx]
            # upsample predictions to the target size
            src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                    mode="bilinear", align_corners=False)
            src_masks = src_masks[:, 0].flatten(1)
            target_masks = target_masks[tgt_idx].flatten(1)

        elif self.model_name == 'dn_detr':
            src_masks = src_masks[src_idx]
            masks = [t["masks"] for t in targets]
            # TODO use valid to mask invalid areas due to padding in loss
            target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
            target_masks = target_masks.to(src_masks)
            target_masks = target_masks[tgt_idx]
            # upsample predictions to the target size
            src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                    mode="bilinear", align_corners=False)
            src_masks = src_masks[:, 0].flatten(1)
            target_masks = target_masks.flatten(1)
            target_masks = target_masks.view(src_masks.shape)


        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, buffer_construct_loss=False, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # for initialization
        self.losses_for_replay = {}
        self.losses_for_replay['loss_bbox'] = []
        self.losses_for_replay['loss_giou'] = []
        self.losses_for_replay['loss_labels'] = []
        if self.model_name == 'dn_detr':
            mask_dict = outputs[1]            
            outputs = outputs[0]

        self.buffer_construct_loss = buffer_construct_loss
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        if return_indices:
            indices0_copy = indices
            indices_list = []        

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_outputs['gt'] = outputs['gt']
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)                
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            enc_outputs['gt'] = outputs['gt']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if self.model_name == 'dn_detr':
            from models.dn_detr.dn_components import compute_dn_loss
            # dn loss computation
            aux_num = 0
            if 'aux_outputs' in outputs:
                aux_num = len(outputs['aux_outputs'])
            dn_losses = compute_dn_loss(mask_dict, self.training, aux_num, self.focal_alpha)
            losses.update(dn_losses)

            if return_indices:
                indices_list.append(indices0_copy)
                return losses, indices_list
        return losses