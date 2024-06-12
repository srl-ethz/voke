###################################################################################################
# Imports
###################################################################################################
import os
import numpy as np
import glob
import shutil
import logging
import time
import sys
sys.path.append("../")
from models.model import ImagePredictionModel

import cv2
import progressbar
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# import CNN.models.vgg as vgg

# from models import resnet, capsnet

from config import args

# from dt_apriltags import Detector

from utils import *
# import scipy.spatial.transform as xform


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

###################################################################################################
# Functions
###################################################################################################

def init_objs():
    objs = AvgrageMeter()
    objs_list = []
    for i in range(0):
        objs_list.append(AvgrageMeter())
    return objs, objs_list


def update_objs(objs, objs_list, loss, loss_list, batch_size):
    objs.update(loss, batch_size)
    for i in range(len(loss_list)):
        objs_list[i].update(loss_list[i], batch_size)
    return objs, objs_list

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


# def findCenter(pathOfImage):
#     #Define click event
#     position = []
#     def click_event(event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             position.append([x,y])
#         if len(position) == 2:
#             cv2.destroyAllWindows()
#
#     # Load image
#     img = cv2.imread(pathOfImage, 0)
#     img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
#     xMax = img.shape[1]
#
#
#
#     # Display image to determine arm location (left-click to choose)
#     imageName = 'Left-click beginning and end of arm'
#     cv2.imshow(imageName, img)
#     cv2.setMouseCallback(imageName, click_event)
#     cv2.waitKey(0)
#
#     # Rotate back
#     tempPosition = np.array(position)
#     position[0][0] = tempPosition[0,1]
#     position[0][1] = xMax - tempPosition[0,0]
#     position[1][0] = tempPosition[1,1]
#     position[1][1] = xMax - tempPosition[1,0]
#
#
#     # Crop image based on selection
#     armCenter = 0.5 * (np.array(position[1]) - np.array(position[0])) + np.array(position[0])
#     armLength = int(np.sqrt((position[0][0] - position[1][0])**2 + (position[0][1] - position[1][1])**2 ) )
#
#     print("Position 0:", position[0])
#     print("Position 1:", position[1])
#     print("Arm Center:", armCenter)
#     print("Arm Length:", int(armLength))
#
#     return [armCenter, armLength]

def rotate(image, angle, center=None):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    if center is None:
        (cX, cY) = (w / 2, h / 2)
    else:
        (cX, cY) = (center[0], center[1])

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def cropImage(img, armCenter, armLength, maxShift, maxRotation, imageSize):
    dirName = os.path.dirname(__file__)
    (h, w) = img.shape[:2]
    rotationAngle = (np.random.uniform()*2 - 1) * maxRotation
    R = cv2.getRotationMatrix2D(tuple(armCenter), rotationAngle, 1.0)
    img = cv2.warpAffine(img, R, (w, h))
    cropSize = int(1.1*armLength + 2*maxShift)
    shift_hor = int(np.random.uniform()*2*maxShift - maxShift)
    shift_vert = int(np.random.uniform()*2*maxShift - maxShift)
    corner = (int(armCenter[0] - cropSize/2 + shift_hor), int(armCenter[1] - cropSize/2 + shift_vert))
    # cv2.imwrite(os.path.join(dirName, "test/00_original.png"), img)

    img_cut = img[corner[1]:(corner[1]+cropSize), corner[0]:(corner[0]+cropSize)].copy()
    # cv2.imwrite(os.path.join(dirName, "test/01_cropped.png"), img_cut)
    
    img_cut = cv2.resize(img_cut, (imageSize, imageSize), interpolation=cv2.INTER_AREA)
    # cv2.imwrite(os.path.join(dirName, "test/02_resized.png"), img_cut)
    return img_cut

def cropImageSoPra(img, imageSize, angle, cx, cy, size):
    # dirName = os.path.dirname(__file__)
    # dirName = os.path.dirname(__file__)

    img = rotate(img, angle)
    img = cv2.copyMakeBorder(img, 400, 400, 400, 400, cv2.BORDER_CONSTANT, None, 0)
    img_cut = img[cy + size:int(cy + 7 * size), cx - 3 * size:cx + 3 * size].copy()
    img_cut = cv2.resize(img_cut, (imageSize, imageSize), interpolation=cv2.INTER_AREA)
    # cv2.imwrite(os.path.join(dirName, "resized.png"), img_cut)
    return img_cut


def processImage(img, medianBlurAmount, erosionKernelSize, dilationKernelSize, iterations):
    dirName = os.path.dirname(__file__)
    img = cv2.medianBlur(img, medianBlurAmount)
    # cv2.imwrite(os.path.join(dirName, "test/03_medBlur.png"), img)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imwrite(os.path.join(dirName, "test/04_otsuThresh.png"), img)
    img = cv2.erode(img, np.ones((erosionKernelSize, erosionKernelSize), np.uint8), iterations=iterations)
    # cv2.imwrite(os.path.join(dirName, "test/05_eroded.png"), img)
    img = cv2.dilate(img, np.ones((dilationKernelSize, dilationKernelSize), np.uint8), iterations=iterations)
    # cv2.imwrite(os.path.join(dirName, "test/06_dilated.png"), img)
    return img

def processImageMask(img, medianBlurAmount):
    dirName = os.path.dirname(__file__)
    img = cv2.medianBlur(img, medianBlurAmount)
    # cv2.imwrite(os.path.join(dirName, "test/03_medBlur.png"), img)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imwrite(os.path.join(dirName, "test/04_otsuThresh.png"), img)
    return img

def processImageGrabCut(img, mask, iterations):
    mask = np.where(mask > 0, 1, 0).astype('uint8')
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (1, 1, img.shape[0], img.shape[1])
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.where(img > 0, 255, 0).astype('uint8')
    return img

def generateSet(validation, indices, goOvers, armCenter_1, armLength_1, armCenter_2, armLength_2, grabcut=False):
    savePath = r'../../3segments_noFeatures/images_processed/grabcut'
    setSize = len(indices)*goOvers
    dataset = np.zeros((setSize, 2, args.image_size, args.image_size), dtype=int)
    bar = progressbar.ProgressBar(maxval=setSize, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    b = 0
    for j in range(goOvers):
        for i in indices:
            bar.update(b+1)

            # Select image
            nameOfImage_1 = "frame{0}_1.png".format(i)
            nameOfImage_2 = "frame{0}_2.png".format(i)
            imagePath_1 = os.path.join(args.data, nameOfImage_1)
            imagePath_2 = os.path.join(args.data, nameOfImage_2)

            # Load color images
            if grabcut:
                img_1_color = cv2.imread(imagePath_1)
                img_2_color = cv2.imread(imagePath_2)

                # Crop and rotate images randomly
                if validation == 0:
                    img_1_color = cropImage(img_1_color, armCenter_1, armLength_1, args.max_shift,
                                            args.max_rotation, args.image_size)
                    img_2_color = cropImage(img_2_color, armCenter_2, armLength_2, args.max_shift,
                                            args.max_rotation, args.image_size)
                else:
                    img_1_color = cropImage(img_1_color, armCenter_1, armLength_1, 0, 0, args.image_size)
                    img_2_color = cropImage(img_2_color, armCenter_2, armLength_2, 0, 0, args.image_size)

                # Load images
                img_1 = cv2.imread(imagePath_1, cv2.IMREAD_GRAYSCALE)
                img_2 = cv2.imread(imagePath_2, cv2.IMREAD_GRAYSCALE)

                # Crop and rotate images randomly
                if validation == 0:
                    img_1 = cropImage(img_1, armCenter_1, armLength_1, args.max_shift, args.max_rotation,
                                      args.image_size)
                    img_2 = cropImage(img_2, armCenter_2, armLength_2, args.max_shift, args.max_rotation,
                                      args.image_size)
                else:
                    img_1 = cropImage(img_1, armCenter_1, armLength_1, 0, 0, args.image_size)
                    img_2 = cropImage(img_2, armCenter_2, armLength_2, 0, 0, args.image_size)

                # Process grayscale image to binary image with GrabCut
                img_1_binary = processImageMask(np.uint8(img_1), args.median_blur_amount)
                img_2_binary = processImageMask(np.uint8(img_2), args.median_blur_amount)
                img_1 = processImageGrabCut(img_1_color, img_1_binary, args.iterations)
                img_2 = processImageGrabCut(img_2_color, img_2_binary, args.iterations)

                # Save processed images
                imgPath_1 = os.path.join(savePath, nameOfImage_1)
                imgPath_2 = os.path.join(savePath, nameOfImage_2)
                cv2.imwrite(imgPath_1, img_1)
                cv2.imwrite(imgPath_2, img_2)

            else:
                # Load images
                img_1 = cv2.imread(imagePath_1, cv2.IMREAD_GRAYSCALE)
                img_2 = cv2.imread(imagePath_2, cv2.IMREAD_GRAYSCALE)

                # Crop and rotate images randomly
                if validation == 0:
                    img_1 = cropImage(img_1, armCenter_1, armLength_1, args.max_shift, args.max_rotation,
                                      args.image_size)
                    img_2 = cropImage(img_2, armCenter_2, armLength_2, args.max_shift, args.max_rotation,
                                      args.image_size)
                else:
                    img_1 = cropImage(img_1, armCenter_1, armLength_1, 0, 0, args.image_size)
                    img_2 = cropImage(img_2, armCenter_2, armLength_2, 0, 0, args.image_size)

                # Process grayscale image to binary image
                img_1 = processImage(np.uint8(img_1), args.median_blur_amount, args.erosionKernelSize,
                                            args.dilationKernelSize, args.iterations)
                img_2 = processImage(np.uint8(img_2), args.median_blur_amount, args.erosionKernelSize,
                                            args.dilationKernelSize, args.iterations)

            dataset[b, 0, :, :] = img_1/255
            dataset[b, 1, :, :] = img_2/255
            
            b += 1
    return dataset


def generateSetSoPrA(indices, label_idx, datapath):
    savePath = r'../../../coding/vise-experiments/vise/export/crop'
    setSize = len(indices)
    dataset = np.zeros((setSize, 2, args.image_size, args.image_size), dtype=int)

    # at_detector = Detector(families='tag36h11',
    #                        nthreads=1,
    #                        quad_decimate=1.0,
    #                        quad_sigma=0.0,
    #                        refine_edges=1,
    #                        decode_sharpening=0.25,
    #                        debug=0)
    #
    # with open('./test_info.yaml', 'r') as stream:
    #     parameters = yaml.load(stream, Loader=yaml.Loader)
    # cameraMatrix = np.array(parameters['K']).reshape((3, 3))
    # camera_params = (cameraMatrix[0, 0], cameraMatrix[1, 1], cameraMatrix[0, 2], cameraMatrix[1, 2])

    bar = progressbar.ProgressBar(maxval=setSize,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    b = 0
    for i in indices:
        bar.update(b + 1)

        # Select image
        nameOfImage_1 = "frame{0}_1.png".format(int(label_idx[i]))
        nameOfImage_2 = "frame{0}_2.png".format(int(label_idx[i]))
        imagePath_1 = os.path.join(datapath, nameOfImage_1)
        imagePath_2 = os.path.join(datapath, nameOfImage_2)

        # Load images
        img_1 = cv2.imread(imagePath_1, cv2.IMREAD_GRAYSCALE)
        img_2 = cv2.imread(imagePath_2, cv2.IMREAD_GRAYSCALE)

        # Rotate and crop images
        img_1 = cropImageSoPra(img_1, args.image_size, 87.28877680765358, 1016, 1283, 138)
        img_2 = cropImageSoPra(img_2, args.image_size, -88.679629624479, 931, 912, 215)

        # img_1 = cropImageSoPra(img_1, args.image_size, 87.00553670445413, 1067, 1025, 134)  # new camera position
        # img_2 = cropImageSoPra(img_2, args.image_size, -87.72196999099371, 961, 923, 215)

        # Process grayscale image to binary image
        img_1 = processImage(np.uint8(img_1), args.median_blur_amount, args.erosionKernelSize,
                             args.dilationKernelSize, args.iterations)
        img_2 = processImage(np.uint8(img_2), args.median_blur_amount, args.erosionKernelSize,
                             args.dilationKernelSize, args.iterations)

        dirName = "./test_image"
        cv2.imwrite(os.path.join(dirName, "img_1.png"), img_1)
        cv2.imwrite(os.path.join(dirName, "img_2.png"), img_2)

        import sys
        sys.exit(1)

        dataset[b, 0, :, :] = img_1 / 255
        dataset[b, 1, :, :] = img_2 / 255

        b += 1
    return dataset


def saveSampleData(datapath):
    nameOfImage_1 = "frame0_1.png"
    nameOfImage_2 = "frame0_2.png"
    imagePath_1 = os.path.join(datapath, nameOfImage_1)
    imagePath_2 = os.path.join(datapath, nameOfImage_2)

    # Load images
    img_1 = cv2.imread(imagePath_1, cv2.IMREAD_GRAYSCALE)
    img_2 = cv2.imread(imagePath_2, cv2.IMREAD_GRAYSCALE)

    # Rotate and crop images
    img_1 = cropImageSoPra(img_1, args.image_size, 87.28877680765358, 1016, 1283, 138)
    img_2 = cropImageSoPra(img_2, args.image_size, -88.679629624479, 931, 912, 215)

    # img_1 = cropImageSoPra(img_1, args.image_size, 87.00553670445413, 1067, 1025, 134)  # new camera position
    # img_2 = cropImageSoPra(img_2, args.image_size, -87.72196999099371, 961, 923, 215)

    # Process grayscale image to binary image
    img_1 = processImage(np.uint8(img_1), args.median_blur_amount, args.erosionKernelSize,
                         args.dilationKernelSize, args.iterations)
    img_2 = processImage(np.uint8(img_2), args.median_blur_amount, args.erosionKernelSize,
                         args.dilationKernelSize, args.iterations)

    dirName = "./test_image"
    cv2.imwrite(os.path.join(dirName, "img_1.png"), img_1)
    cv2.imwrite(os.path.join(dirName, "img_2.png"), img_2)

def generateBatches(dataset, indices, batchSize, goOvers, label):
    dataset = torch.FloatTensor(dataset)
    dataset = TensorDataset(dataset, torch.cat(goOvers*[label[indices, :]]))
    batches = DataLoader(dataset, batch_size=batchSize, shuffle=True)
    return batches

def generateBatchesSoPrA(dataset, indices, batchSize, goOvers, label):
    dataset = torch.FloatTensor(dataset)
    dataset = TensorDataset(dataset, torch.cat([label[indices, 1:7]]))
    batches = DataLoader(dataset, batch_size=batchSize, shuffle=True)
    return batches

def defineSet():
    datasetIndices = np.arange(args.max_dataset_size)
    np.random.shuffle(datasetIndices)
    datasetIndices = datasetIndices[:args.dataset_size]
    splitLocation = round(args.dataset_size*args.validation_set_size)
    trainingSetIndices = datasetIndices[splitLocation:]
    validationSetIndices = datasetIndices[:splitLocation]
    return trainingSetIndices, validationSetIndices


def defineSetSoPrA(label):
    datasetIndices = np.arange(len(label))
    np.random.shuffle(datasetIndices)
    datasetIndices = datasetIndices[:args.dataset_size]
    splitLocation1 = round(args.dataset_size*args.validation_set_size)
    splitLocation2 = round(args.dataset_size * (args.validation_set_size + args.testing_set_size))
    trainingSetIndices = datasetIndices[splitLocation2:]
    validationSetIndices = datasetIndices[:splitLocation1]
    testingSetIndices = datasetIndices[splitLocation1:splitLocation2]
    return trainingSetIndices, validationSetIndices, testingSetIndices

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def initLog():
    # Create log directories
    make_dir(args.log_dir)
    args.log_dir = os.path.join(args.log_dir, time.strftime("%Y%m%d-%H%M%S-" + args.network))
    create_exp_dir(args.log_dir, scripts_to_save=glob.glob('*.py'))
    make_dir(os.path.join(args.log_dir, 'weights'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.log_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("args = %s", args)

    return args.log_dir, logging

def initLog_result(result_dir, model):
    # Create log directories
    make_dir(result_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(result_dir, args.network + '.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("Log dir = %s", args.log_dir)
    logging.info('Model:  %s Parameter count: %e MB', args.network, count_parameters_in_MB(model))
    return logging


def initNet(comment, learning_rate, momentum):
    cfg = {
        'model':{
            'in_channels': 2,
            'num_classes': args.num_classes,
            'dropout': args.dropout,
            'type_model': args.network,
        },
        'data':{
            'size': 256,
        },
    }
    model = ImagePredictionModel(cfg)
    # tb = SummaryWriter(comment=comment)
    # if args.network == "vgg_s_bn":
    #     model = vgg.Net_bn_short(p_dropout)
    # elif args.network == "vgg11_bn":
    #     model = vgg.vgg11_bn(num_classes=6, in_channels=2)
    # elif args.network == 'resnet18':
    #     model = resnet.resnet18(num_classes=6, in_channels=2)
    # elif args.network == 'resnet50':
    #     model = resnet.resnet50(num_classes=6, in_channels=2)
    # elif args.network == 'effnetv2_s':
    #     model = efficientnetv2.effnetv2_s(num_classes=6, in_channels=2)
    # elif args.network == 'capsnet':
    #     model = capsnet.EfficientCapsNet(num_classes=6, in_channels=2)
    # model.load_state_dict(torch.load('C:/Users/Enceladus/OneDrive/ETH/01_Semester Project/Coding/Qualisys/Code/3D/models/21-02-09__15-05-07__LR=0.01_M=0.9_BS=32_DSS=5000_Overs=1_shift=0_rot=0_img=256_BN=1_Net=Net2_drop=0.5.pt', map_location=torch.device('cpu')))
    model.to(args.device)
    #optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=momentum, weight_decay=1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # criterion = torch.nn.L1Loss(reduction='sum')  # MAELoss
    if args.loss_type == 1:
        criterion = torch.nn.L1Loss()  # MAELoss
    elif args.loss_type == 2:
        criterion = torch.nn.MSELoss()  # MSELoss
    return model, optimizer, criterion  #, tb

def trainModel(model, optimizer, getLoss, trainingBatches, trainingSetSize, logging):
    model.train()

    objs = AvgrageMeter()

    batch_idx = 0

    for xb, yb in trainingBatches:
        # Push input through network & Evaluate loss function
        output = model(xb.to(args.device))
        del xb
        loss = getLoss(output, yb)
        # lossSum += loss.item()

        # error = torch.abs(output - yb)
        # errorSum += torch.sum(error, axis=0)

        # Backpropagate output & update weights
        model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        objs.update(loss.item(), output.size(0))

        if batch_idx % args.report_freq == 0:
            logging.info('train %03d %e', batch_idx, objs.avg)
        batch_idx += 1
    
    # lossSum /= trainingSetSize
    # errorSum /= trainingSetSize

    return model, objs.avg  #, errorSum

def validateModel(model, getLoss, validationBatches, validationSetSize, logging):
    model.eval()
    objs = AvgrageMeter()
    batch_idx = 0
    with torch.no_grad():
        # lossSum = 0
        # errorSum = 0
        L2_sum = np.zeros([2])
        L2_max = np.zeros([2])

        for xb, yb in validationBatches:
            output = model(xb.to(args.device))
            new_L2_sum, new_L2_max = evalError(output, yb)
            L2_sum += new_L2_sum

            if L2_max[0] < new_L2_max[0]:
                L2_max[0] = new_L2_max[0]
            if L2_max[1] < new_L2_max[1]:
                L2_max[1] = new_L2_max[1]

            loss = getLoss(output, yb)
            objs.update(loss.item(), output.size(0))
            # lossSum += loss.item()

            # error = torch.abs(output - yb)
            # errorSum += torch.sum(error, axis=0)
            if batch_idx % args.report_freq == 0:
                logging.info('valid %03d %e', batch_idx, objs.avg)
            batch_idx += 1

    L2_mean = L2_sum / validationSetSize
    # lossSum /= validationSetSize
    # errorSum /= validationSetSize

    return L2_mean, L2_max, objs.avg  #, errorSum

def testModel(model, labels_mean, labels_std, validationBatches, validationSetSize, logging):
    model.eval()
    count = 0
    total_forward_time = 0
    start_forward_time = time.time()
    with torch.no_grad():
        # lossSum = 0
        # errorSum = 0
        prediction = np.zeros((validationSetSize, args.num_classes))
        target = np.zeros((validationSetSize, args.num_classes))

        for xb, yb in validationBatches:
            output = model(xb.to(args.device))
            total_forward_time += time.time() - start_forward_time
            prediction[count:count+output.size(0)] = output.cpu().detach().numpy() * labels_std + labels_mean
            target[count:count+output.size(0)] = yb.cpu().detach().numpy() * labels_std + labels_mean
            count += output.size(0)
            start_forward_time = time.time()

        error = target - prediction
        L2_sum = np.zeros([error.shape[0], 2])
        for i in range(error.shape[0]):
            L2_sum[i, :] = np.array([np.sqrt(error[i, 0] ** 2 + error[i, 1] ** 2 + error[i, 2] ** 2), 
                                     np.sqrt(error[i, 3] ** 2 + error[i, 4] ** 2 + error[i, 5] ** 2)])
        error_mean = np.mean(L2_sum, axis=0)
        error_std = np.std(L2_sum, axis=0)
        error_max = np.max(L2_sum, axis=0)

    return error_mean, error_std, error_max, total_forward_time  #, errorSum


def evalError(prediction, label):
    prediction = prediction.cpu().detach().numpy()
    label = label.cpu().detach().numpy()

    #labels_mean = np.loadtxt('labels/3_labels_mean.txt', dtype='float', delimiter=',')
    labels_std = np.loadtxt(os.path.join(args.label_folder, args.label_std_filename), dtype='float', delimiter=',')

    error = abs(prediction - label)
    error = error*labels_std

    L2_sum = np.zeros([error.shape[0], 2])
    for i in range(error.shape[0]):
        L2_sum[i, :] = np.array([np.sqrt(error[i, 0] ** 2 + error[i, 1] ** 2 + error[i, 2] ** 2),
                                 np.sqrt(error[i, 3] ** 2 + error[i, 4] ** 2 + error[i, 5] ** 2)])

    L2_max = np.max(L2_sum, axis=0)
    L2_sum = np.sum(L2_sum, axis=0)

    return L2_sum, L2_max

# def updateTensorboard(tb, epoch, train_loss, valid_loss,train_error, valid_error, L2_mean, L2_max, model, validationBatches):
def updateTensorboard(tb, epoch, train_loss, valid_loss, L2_mean, L2_max, model,
                          validationBatches):
    tb.add_scalar("Loss/Training", train_loss[epoch], epoch)
    tb.add_scalar("Loss/Validation", valid_loss[epoch], epoch)

    # tb.add_scalar("Error/Training/P1_x", train_error[epoch][0], epoch)
    # tb.add_scalar("Error/Training/P1_y", train_error[epoch][1], epoch)
    # tb.add_scalar("Error/Training/P1_z", train_error[epoch][2], epoch)
    # tb.add_scalar("Error/Training/P2_x", train_error[epoch][3], epoch)
    # tb.add_scalar("Error/Training/P2_y", train_error[epoch][4], epoch)
    # tb.add_scalar("Error/Training/P2_z", train_error[epoch][5], epoch)
    # tb.add_scalar("Error/Training/P3_x", train_error[epoch][6], epoch)
    # tb.add_scalar("Error/Training/P3_y", train_error[epoch][7], epoch)
    # tb.add_scalar("Error/Training/P3_z", train_error[epoch][8], epoch)
    #
    # tb.add_scalar("Error/Validation/P1_x", valid_error[epoch][0], epoch)
    # tb.add_scalar("Error/Validation/P1_y", valid_error[epoch][1], epoch)
    # tb.add_scalar("Error/Validation/P1_z", valid_error[epoch][2], epoch)
    # tb.add_scalar("Error/Validation/P2_x", valid_error[epoch][3], epoch)
    # tb.add_scalar("Error/Validation/P2_y", valid_error[epoch][4], epoch)
    # tb.add_scalar("Error/Validation/P2_z", valid_error[epoch][5], epoch)
    # tb.add_scalar("Error/Validation/P3_x", valid_error[epoch][6], epoch)
    # tb.add_scalar("Error/Validation/P3_y", valid_error[epoch][7], epoch)
    # tb.add_scalar("Error/Validation/P3_z", valid_error[epoch][8], epoch)

    tb.add_scalar("L2_Mean/P1", L2_mean[0], epoch)
    tb.add_scalar("L2_Mean/P2", L2_mean[1], epoch)
    # tb.add_scalar("L2_Mean/P3", L2_mean[2], epoch)
    tb.add_scalar("L2_Max/P1", L2_max[0], epoch)
    tb.add_scalar("L2_Max/P2", L2_max[1], epoch)
    # tb.add_scalar("L2_Max/P3", L2_max[2], epoch)
    tb.flush()
    return


