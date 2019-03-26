import numpy as np
from .. import constants


class Drawings:
    SpaceInvaders = np.array(
        [[0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0],
         [0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0],
         [0, 0, 4, 4, 4, 4, 4, 4, 4, 0, 0],
         [0, 4, 4, 0, 4, 4, 4, 0, 4, 4, 0],
         [4, 0, 4, 4, 4, 4, 4, 4, 4, 0, 4],
         [4, 0, 4, 0, 0, 0, 0, 0, 4, 0, 4],
         [0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0]], dtype=int)

    Frog = np.array(
        [[0, 2, 2, 2, 0, 2, 2, 2, 0],
         [2, 2, 1, 4, 2, 2, 1, 4, 2],
         [2, 2, 2, 2, 2, 2, 2, 2, 2],
         [2, 2, 2, 2, 2, 2, 2, 2, 0],
         [2, 2, 2, 4, 4, 4, 4, 4, 4],
         [2, 2, 4, 4, 4, 4, 4, 4, 4],
         [3, 2, 2, 2, 2, 2, 2, 2, 0]], dtype=int)

    Batman = np.array(
        [[0, 0, 4, 0, 0, 0, 0, 4, 0, 0],
         [0, 0, 4, 0, 0, 0, 0, 4, 0, 0],
         [0, 0, 4, 4, 4, 4, 4, 4, 4, 4],
         [0, 0, 4, 4, 4, 4, 4, 4, 4, 4],
         [0, 0, 4, 4, 2, 4, 4, 4, 4, 2],
         [0, 0, 4, 4, 4, 4, 4, 4, 4, 4],
         [0, 0, 4, 4, 1, 1, 1, 1, 1, 1]], dtype=int)

    BatSignal = np.array(
        [[0, 0, 1, 4, 1, 1, 1, 0, 0, 0],
         [0, 1, 4, 4, 4, 1, 4, 1, 0, 0],
         [0, 4, 4, 4, 4, 4, 4, 4, 1, 0],
         [0, 4, 1, 4, 4, 4, 4, 1, 1, 0],
         [0, 0, 1, 1, 4, 4, 4, 4, 1, 0],
         [0, 0, 1, 4, 1, 4, 4, 4, 4, 0],
         [0, 0, 1, 1, 1, 1, 4, 4, 1, 0]], dtype=int)

    BatSignal2 = np.array(
        [[0, 0, 4, 1, 1, 1, 1, 4, 0, 0],
         [0, 4, 1, 1, 4, 4, 1, 1, 4, 0],
         [0, 4, 4, 4, 4, 4, 4, 4, 4, 0],
         [0, 4, 4, 1, 4, 4, 1, 4, 4, 0],
         [0, 4, 1, 1, 4, 4, 1, 1, 4, 0],
         [0, 0, 4, 1, 1, 4, 1, 4, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=int)

    def getCalendar():
        blk = np.zeros((7, 21), dtype=int)
        res = np.concatenate((blk, Drawings.SpaceInvaders), axis=1)
        res = np.concatenate((res, blk), axis=1)
        return res.T.reshape(constants.FLATSHAPE)

    def getCalendar2():
        blk3 = np.zeros((7, 3), dtype=int)
        blk1 = np.zeros((7, 1), dtype=int)
        res = np.concatenate((blk3, Drawings.SpaceInvaders), axis=1)
        res = np.concatenate((res, blk1), axis=1)
        res = np.concatenate((res, Drawings.SpaceInvaders), axis=1)
        res = np.concatenate((res, blk1), axis=1)
        res = np.concatenate((res, Drawings.SpaceInvaders), axis=1)
        res = np.concatenate((res, blk1), axis=1)
        res = np.concatenate((res, Drawings.SpaceInvaders), axis=1)
        res = np.concatenate((res, blk3), axis=1)
        return res.T.reshape(constants.FLATSHAPE)

    def getCalendar3():
        blk1 = np.zeros((7, 44), dtype=int)
        res = np.concatenate((Images.Frog, blk1), axis=1)
        return res.T.reshape(constants.FLATSHAPE)

    def getCalendar4():
        blk1 = np.zeros((7, 33), dtype=int)
        res = np.concatenate((Drawings.Batman, blk1), axis=1)
        res = np.concatenate((res, Drawings.BatSignal2), axis=1)
        return res.T.reshape(constants.FLATSHAPE)
