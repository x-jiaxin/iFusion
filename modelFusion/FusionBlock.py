"""
@Author: 幻想
@Date: 2022/05/02 15:49
"""
from utils.fusion import fusion


class FusionBlock:
    def __init__(self) -> None:
        super().__init__()
        self.fS = None
        self.fT = None
        self.fusionBlock = fusion()

    def doFusion(self, *args):
        # return S, T 1024
        fS, fT = args
        fSNew, fTNew = self.fusionBlock(fS, fT)
        return fSNew, fTNew

    def getFeatureT(self, args):
        self.fT = args

    def getFeatureS(self, args):
        self.fS = args
