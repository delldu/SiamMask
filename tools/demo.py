# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from tools.test import *
import pdb

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    from models.siammask_sharp import SiameseTracker

    siammask = SiameseTracker()
    # torch.jit.script(siammask)

    if args.resume:
        # assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # torch.jit.script(siammask)
    # pdb.set_trace()

    # Parse Image file
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    ims = [cv2.imread(imf) for imf in img_files]

    # ims[0].shape -- (480, 854, 3), 'numpy.ndarray', Range: 0-255, uint8, BGR format ?
    # ims = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in ims]

    # Select ROI
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # try:
    #     init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
    #     x, y, w, h = init_rect
    # except:
    #     exit()
    x, y, h, w = 300, 100, 280, 180
    target_pos = np.array([x + w / 2, y + h / 2])
    target_size = np.array([w, h])
    state = TrackingStart(siammask, ims[0], target_pos, target_size, device=device)  # init tracker

    toc = 0
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        state = TrackingDoing(siammask, state, im, mask_enable=True, device=device)  # track
        # (Pdb) pp state.keys() -- dict_keys('avg_chans', 'target_pos', 'target_size', 'score', 'mask')

        # location = state['ploygon'].flatten()
        mask = state['mask']

        # BGR format !!!
        im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
        # cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
        
        cv2.imshow('SiamMask', im)
        key = cv2.waitKey(1)
        if key > 0:
            break
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
