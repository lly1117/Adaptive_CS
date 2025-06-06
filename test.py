import os, glob, random, torch, cv2
import numpy as np
from argparse import ArgumentParser
from model_test import Adaptive_CS
from utils import *
from skimage.metrics import structural_similarity as ssim

parser = ArgumentParser(description='Adaptive_CS')
parser.add_argument('--epoch', type=int, default=320)
parser.add_argument('--phase_num', type=int, default=13)
parser.add_argument('--block_size', type=int, default=8)
parser.add_argument('--model_dir', type=str, default='model')
parser.add_argument('--high_rate', type=float, default=0.5)
parser.add_argument('--low_rate', type=float, default=0.01)
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--testset_name', type=str, default='testset')
parser.add_argument('--result_dir', type=str, default='test_out')
parser.add_argument('--log_dir', type=str, default='log')
parser.add_argument('--gpu_list', type=str, default='0')

args = parser.parse_args()
epoch = args.epoch
N_p = args.phase_num
B = args.block_size

log_path = '%s/%s_test_layer_%d_block_%d.txt' % (args.log_dir, args.data_dir, N_p, B)
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fixed seed for reproduction
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

img_nf = 1  # image channel number
N = B * B
#cs_ratio_list = [0.01, 0.04, 0.10, 0.25, 0.30, 0.40, 0.50]  # ratios in [0, 1] are all available
cs_ratio_list = [args.low_rate]

# create and initialize Adaptive_CS
model = Adaptive_CS(N_p, B, img_nf, torch.zeros(N, N))
model = torch.nn.DataParallel(model).to(device)
model_dir = '%s/layer_%d_block_%d' % (args.model_dir, N_p, B)
model.load_state_dict(torch.load('%s/net_params_%d.pkl' % (model_dir, epoch), map_location=device))
#result_dir = '%s/%.2f_%.2f' % (args.result_dir, args.high_rate, args.low_rate)
rate_dir = '%.2f_%.2f' % (args.high_rate, args.low_rate)

# test set info
test_image_paths = glob.glob(os.path.join(args.data_dir, args.testset_name) + '/*')
test_image_num = len(test_image_paths)

output_dir = os.path.join(args.result_dir, args.testset_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def test(cs_ratio, epoch_num, rand_modes):
    with torch.no_grad():
        PSNR_list, SSIM_list = [], []
        for i in range(test_image_num):
            image_path = test_image_paths[i]
            test_image = cv2.imread(test_image_paths[i], 1)  # read test data from image file
            test_image_ycrcb = cv2.cvtColor(test_image, cv2.COLOR_BGR2YCrCb)
            
            img, old_h, old_w, img_pad, new_h, new_w = my_zero_pad(test_image_ycrcb[:,:,0])
            img_pad = img_pad.reshape(1, 1, new_h, new_w) / 255.0  # normalization
            
            x_input = torch.from_numpy(img_pad)
            x_input = x_input.type(torch.FloatTensor).to(device)

            #x_output = model(x_input, int(np.ceil(cs_ratio * N)), rand_modes)
            x_output, cs = model(x_input, int(np.ceil(cs_ratio * N)), rand_modes, args.high_rate, args.low_rate)
            x_output = x_output.cpu().data.numpy().squeeze()
            x_output = np.clip(x_output[:old_h, :old_w], 0, 1).astype(np.float64) * 255.0

            PSNR = psnr(x_output, img)
            SSIM = ssim(x_output, img, data_range=255)

            # print('[%d/%d] %s, PSNR: %.2f, SSIM: %.4f' % (i, test_image_num, image_path, PSNR, SSIM))

            # save restored image into files
            test_image_ycrcb[:, :, 0] = x_output
            test_image_rgb = cv2.cvtColor(test_image_ycrcb, cv2.COLOR_YCrCb2BGR).astype(np.uint8)

            save_path_prefix = image_path.replace(args.data_dir, args.result_dir)
            #cv2.imwrite('%s_Adaptive_CS_ratio_%.2f_epoch_%d_PSNR_%.2f_SSIM_%.4f.png' % (save_path_prefix, cs_ratio, epoch_num, PSNR, SSIM), test_image_rgb)
            cv2.imwrite('%s' % (save_path_prefix), test_image_rgb)

            PSNR_list.append(PSNR)
            SSIM_list.append(SSIM)

    return float(np.mean(PSNR_list)), float(np.mean(SSIM_list)), cs
    
    
test_time = 1
for cs_ratio in cs_ratio_list:
    avg_psnr, avg_ssim = 0.0, 0.0
    avg_cs = 0.0
    for i in range(test_time):
        rand_modes = [random.randint(0, 7) for _ in range(N_p)]  # randomly choose a transformation for each phase
        cur_psnr, cur_ssim, cur_cs = test(cs_ratio, epoch, rand_modes)
        avg_psnr += cur_psnr
        avg_ssim += cur_ssim
        avg_cs += cur_cs
    avg_psnr /= test_time
    avg_ssim /= test_time
    avg_cs /= test_time
    print('%.2f   %.2f   %.4f' % (avg_psnr, avg_ssim, avg_cs))
    #print('CS ratio is %.2f, avg PSNR is %.2f, avg SSIM is %.4f.' % (cs_ratio, avg_psnr, avg_ssim))
    log_data = 'High ratio is %.2f , low ratio is %.2f , avg PSNR is %.2f , avg SSIM is %.2f , avg cs is %.4f , data_dir = %s , testset_name = %s .' % (args.high_rate, args.low_rate, avg_psnr, avg_ssim, avg_cs, args.data_dir, args.testset_name)
    with open(log_path, 'a') as log_file:
        log_file.write(log_data + '\n')
