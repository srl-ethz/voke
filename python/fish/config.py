import argparse
import torch

parser = argparse.ArgumentParser("vise_fish")

parser.add_argument('--device', type=str, default='cuda', help='device, cuda or cpu')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--dataset_folder', type=str, required=True, help='dataset folder path')
parser.add_argument('--label_folder', type=str, required=True, help='label folder path')
parser.add_argument('--label_mean_filename', type=str, default='labels_mean.txt', help='label file name')
parser.add_argument('--label_std_filename', type=str, default='labels_std.txt', help='label file name')

parser.add_argument('--network', type=str, default='vgg_s_bn', help='type of CNN network')
parser.add_argument('--num_classes', type=int, default=9, help='number of outputs')

parser.add_argument('--batch_size', type=int, default=24, help='batch size')
parser.add_argument('--max_dataset_size', type=int, default=300, help='maximum dataset size')
parser.add_argument('--dataset_size', type=int, default=300, help='dataset size')
parser.add_argument('--validation_set_size', type=float, default=0.2, help='validation dataset proportion')
parser.add_argument('--testing_set_size', type=float, default=0, help='testing dataset proportion')
parser.add_argument('--training_go_overs', type=int, default=60, help='training go overs')
parser.add_argument('--validation_go_overs', type=int, default=60, help='validation go overs')

# Learning
parser.add_argument('--learning_rate', type=float, default=1e-4, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--valid_freq', type=float, default=1, help='validation frequency')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--loss_type', type=int, default=1, help='1 for MAE and 2 for MSE loss')

parser.add_argument('--scheduler_active', action='store_false', default=True, help='use lr scheduler')
parser.add_argument('--scheduler_type', type=str, default='step', help='type of lr scheduler type')
parser.add_argument('--scheduler_once', action='store_true', default=False, help='use lr scheduler only once')
parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='lr scheduler gamma')
parser.add_argument('--scheduler_step_size', type=int, default=30, help='StepLR scheduler step size')
parser.add_argument('--scheduler_cos_epoch', type=int, default=300, help='T_max for cosine lr scheduler')

# Image Crop
parser.add_argument('--image_size', type=int, default=256, help='cropped image size')
parser.add_argument('--max_shift', type=int, default=0, help='maximum shift')
parser.add_argument('--max_rotation', type=int, default=0, help='maximum rotation')

# Image Processing
parser.add_argument('--median_blur_amount', type=int, default=7, help='median blur amount')
parser.add_argument('--erosionKernelSize', type=int, default=3, help='erosion kernel size')
parser.add_argument('--dilationKernelSize', type=int, default=3, help='dilation kernel size')
parser.add_argument('--iterations', type=int, default=3, help='image process iterations')

parser.add_argument('--train_occlusion', action='store_true', default=False, help='')

parser.add_argument('--occlu_width', type=int, default=3, help='image process iterations')
parser.add_argument('--fix_occlu_pos', type=int, default=100, help='image process iterations')
parser.add_argument('--random_insert', action='store_true', default=False, help='use lr scheduler only once')

parser.add_argument('--log_dir', default='checkpoints/', help='Log dir [default: log]')
parser.add_argument('--result_dir', default='../results/fish', help='Result dir [default: log]')
parser.add_argument('--report_freq', type=int, default=10, help='report batch frequency')
parser.add_argument('--wnb_project', type=str, default="vise_cnn_fish", help='W&B project name')
parser.add_argument('--wnb_name', type=str, default='', help='W&B run name')

parser.add_argument('--long_run', default=False, action="store_true")
parser.add_argument('--save', type=str, default='EXP', help='experiment name')


args = parser.parse_args()
