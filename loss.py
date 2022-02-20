import torch
from torch import nn
import torch.nn.functional as F
from Models.layer import VGG19Features

class VggLoss(nn.Module):
    def __init__(self, features = VGG19Features):
        self.features = features
        self.GuidedRegressLossFeatures = ['r{}1'.format(i) for i in range(1, 3)]
        self.FeatMatchLossFeatures = ['r{}1'.format(i) for i in range(1, 6)]
        
        # For GeoAlignLoss
        self.u_val, self.v_val = torch.meshgrid(torch.arange(-1, 1, 1/8), torch.arange(-1, 1, 1/8))
        self.u_val, self.v_val = self.u_val.cuda(), self.v_val.cuda()
        
    def WeightL(self, layer):
        shape = layer.shape
        return 1e3/(shape[1]*shape[1]*shape[1]*shape[2]*shape[3])
        
    def SelfGuidedRegressionLoss(self, gen, out, guidanceMask):
        """Find L1 loss of the sum of [product of Ml{guidance}.(Phi{Igt} - Phi{Iout})].Wl"""
        """Insert Code Here"""

    def GeometricalAlignmentConstraint(self, genR, outR, ):
        """Find L2 loss {[Cku, Ckv] - [Ck'u, Ck'v]}""" 
        """Insert Code Here"""
        
    def FeatureMatchingLoss(self, gen, out):
        """Find L1 loss of sum of Wl.{Phi[Igt] - Phi[Iout]}""" 
        """Insert Code Here"""
           
    def forward(self, gen, out, guidanceMask):
        
        Loss = {}
        
        Loss['SGRL'] = self.SelfGuidedRegressionLoss(gen, out, guidanceMask)
        Loss['GAC'] = self.GeometricalAlignmentConstraint(gen['r41'], out['r41'])
        Loss['FML'] = self.FeatMatchLossFeatures(gen, out)
        
        return Loss