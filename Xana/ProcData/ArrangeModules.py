import numpy as np
import sys


def arrange_cspad_tiles(
    img, ntiles=32, tile_size=(185, 388), inverse=False, common_mode=False
):
    ffg = 0
    ffg2 = 0
    tile_location = np.array(
        [
            [2, 2],
            [3, 2],
            [0, 2],
            [0, 3],
            [1, 0],
            [0, 0],
            [2, 0],
            [2, 1],
            [4, 2],
            [4, 3],
            [5, 0],
            [4, 0],
            [6, 1],
            [6, 0],
            [7, 2],
            [6, 2],
            [5, 4],
            [4, 4],
            [6, 5],
            [6, 4],
            [6, 6],
            [7, 6],
            [4, 7],
            [4, 6],
            [2, 5],
            [2, 4],
            [2, 6],
            [3, 6],
            [0, 6],
            [0, 7],
            [0, 4],
            [1, 4],
        ]
    )

    tile_location2 = np.array(
        [
            [419, 547],
            [627, 542],
            [0, 538],
            [-2, 751],
            [213, 119],
            [2, 118],
            [428, 134],
            [429, 347],
            [830, 508],
            [831, 721],
            [1047, 87],
            [835, 87],
            [1262, 297],
            [1261, 84],
            [1450, 516],
            [1236, 517],
            [1091, 916],
            [879, 915],
            [1306, 1132],
            [1308, 919],
            [1298, 1345],
            [1512, 1345],
            [882, 1534],
            [883, 1321],
            [455, 1178],
            [458, 960],
            [449, 1391],
            [660, 1391],
            [28, 1384],
            [28, 1597],
            [47, 962],
            [259, 962],
        ]
    )

    # tile_offset = np.array([[0, 0], [6, -15], [ 3, -1], [16, -3]])
    quad_offset = np.array(
        [[10 + ffg2, 0 - ffg], [11 - ffg, -10], [3, -1 - ffg2], [16 + ffg2, -3 - ffg2]]
    )
    isUD = np.array(
        [
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        dtype=np.bool,
    )
    isLR = np.array(
        [
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
        ],
        dtype=np.bool,
    )
    isTran = np.array(
        [
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            0,
        ],
        dtype=np.bool,
    )

    def get_slice(i):
        quad_ind = int(np.floor(i / 8))
        tile_location_final = tile_location2[i] + [50, -20] + quad_offset[quad_ind]
        idx1 = slice(
            tile_location_final[0], tile_location_final[0] + tile_size[isTran[i]]
        )
        idx2 = slice(
            tile_location_final[1], tile_location_final[1] + tile_size[~isTran[i]]
        )
        return idx1, idx2

    # This is the arrangement of the 32 modules on the detector chip!
    if inverse:
        arr = np.empty((ntiles, *tile_size), dtype=np.float32)
    else:
        arr = np.zeros((1800, 1800), dtype=np.float32)

    # ATTENTION: There is no final evaluation of all modules and their alignment!!!
    for i in np.arange(ntiles):

        idx1, idx2 = get_slice(i)

        if inverse:
            buffer = img[idx1, idx2]
        else:
            buffer = img[i].copy()

        if common_mode and not inverse:
            buffer -= common_mode(buffer)
        if isTran[i]:
            buffer = np.transpose(buffer)
        if isLR[i]:
            buffer = np.fliplr(buffer)
        if isUD[i]:
            buffer = np.flipud(buffer)

        if inverse:
            arr[i] = buffer
        else:
            arr[idx1, idx2] = buffer

    # image = np.transpose(image)
    # im_blank = commonmode.commonmode(im_blank,commonmode.mask,searchoffset=200,COMrad=3)
    return arr
