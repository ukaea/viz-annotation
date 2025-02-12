"""
@authors: 
    Niraj Bhujel, SciML-STFC-UKRI (niraj.bhujel@stfc.ac.uk)
    Samuel Jackson UKAEA (samuel.jackson@ukaea.uk)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class BCELoss(nn.Module):
	def __init__(self, cfg, ):
		super().__init__()

		self.label_smoothing = cfg['label_smoothing']
		
		if cfg['class_weights'] is not None:
			print(f"BCELoss with class weights: {cfg['class_weights']}")
			self.cls_weights = torch.tensor(cfg['class_weights'])
			reduction = 'none'
		else:
			reduction = 'mean'
			
			self.cls_weights = None

		self.func = nn.BCELoss(reduction=reduction)

	def forward(self, y_pred, y_true):

		assert y_pred.shape==y_true.shape, f"{y_pred.shape=} is not equal to {y_true.shape=}"
		
		y_pred = F.softmax(y_pred, dim=1)

		if self.label_smoothing>0:
			y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / 2
			
		loss = self.func(y_pred, y_true)

		if self.cls_weights is not None:
			weight = torch.ones_like(loss)
			weight[y_true.argmax(-1)==0] = self.cls_weights[0]
			weight[y_true.argmax(-1)==1] = self.cls_weights[1]

			loss = (weight*loss).mean()

		return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        loss = self.loss_fn(logits, targets)
        pt = torch.exp(-loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * loss
        return focal_loss.mean()


class SCLLoss(nn.Module):
	"""
	Supervised contrastive loss
	Multi-Positive Contrastive Loss: https://arxiv.org/pdf/2306.00984.pdf
	Implmentation adapted from: https://github.com/google-research/syn-rep-learn/blob/main/StableRep/models/losses.py#L49
	"""

	def __init__(self, cfg, device):
		super().__init__()
		self.device = device
		self.temperature = cfg['temperature']
		self.logits_mask = None
		self.mask = None

	def set_temperature(self, temp=0.1):
		self.temperature = temp
  
	def compute_cross_entropy(self, p, q):
		"""
		Calculate CE loss i.e., LogSoftmax followed by NNL loss

		Args:
			p (Tensor): [B,B]
			q (Tensor): [B,B]

		Returns:
			Tensor: [B]: Loss
		"""
		
		#Calc log probs: Convert range from [-inf, 0] to [0,1] (using softmax) and then to [-inf,0] (using log)
		q = F.log_softmax(q, dim=-1)
		# p range: [0,1], q range: [-inf, 0], loss range: [-inf, 0]
		# calc NNL loss: will mask out elems in q with non-matching labels (i.e. 0 elems in p) before summing 
		loss = torch.sum(p * q, dim=-1)
		# score for matches with the same class label 
		return - loss.mean()


	def stablize_logits(self, logits):
		logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
		logits = logits - logits_max.detach()
		return logits

	def compute(self, feats, labels):
		# call forward
		with torch.no_grad():
			return self(feats, labels)

	def forward(self, feats, labels):
		"""
		Forward pass: Calculate contrastive loss of anchor images aganist rest of batch (which can contain 
		both positive and negative samples).

		Args:
			feats (Tensor: [B, D]): Feature embedding of inputs. 
		   labels (Tensor: [B, n_classes]): One hot encoding (OHE) for labels idx 0 = undamaged, idx 1 = damaged.

		Returns:
			Tensor: Scalar loss
		"""
		# Get class label from OHE. Shape: [B]
		labels = labels.argmax(-1)
  
		# L2 norm of features so they are within a hypersphere. Shape: [B, D]
		feats = F.normalize(feats, dim=-1, p=2)
  
		# Create a matrix mask by doing a comparison of each label element to all the labels in the batch. 
		# Equivalent to doing XNOR(labels, labels.T)
		# Matching labels (i.e., (0,0) or (1,1)) are set to 1. Different labels ((0,1) and (1,0)) are set to 0.
		# Shape: [B,B]
		mask = torch.eq(labels.view(-1, 1), labels.contiguous().view(1, -1)).float().to(self.device)

		# create a mask to ignore the diagonal elements (i.e., self-similarities). 
		# Shape: [B,B]
		self.logits_mask = torch.scatter(
			torch.ones_like(mask),
			1,
			torch.arange(mask.shape[0]).view(-1, 1).to(self.device),
			0
		)

		# mask out self-similarities. # Shape: [B,B]
		self.mask = mask * self.logits_mask
		mask = self.mask

		# compute logits
		# Shape: [B,D] @ [D,B] / self.temp = [B,B]
		logits = torch.matmul(feats, feats.T) / self.temperature
		# Sets +ve = logits and -ve ~= -inf
		logits = logits - (1 - self.logits_mask) * 1e9

		# (optional): minus the largest logit to stablize logits
		# Shape: [B,B]
		logits = self.stablize_logits(logits)

		# compute ground-truth distribution
		# normalize mask per rows (i.e., spread the probability over the 1's). Shape: [B,B]
		p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
  
		# CE loss on p (matching labels distribution) and logits (feature similarity)
		# Shape: [B]
		loss = self.compute_cross_entropy(p, logits)
		return loss
