###################################################################################################
# Imports
###################################################################################################
import os
import numpy as np
import torch
import functions as f
import pickle
from config import args
import random

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def main():
    # Import labels
    labelPath = os.path.join(args.label_folder, args.label_filename)
    label = np.loadtxt(labelPath, dtype='float', delimiter=',')
    label_idx = label[:, 0]
    label_idx = [int(x) for x in label_idx]
    label = torch.FloatTensor(label).to(args.device)

    # Create training and validation indices to select from dataset
    trainingSetIndices, validationSetIndices = f.defineSet()

    # Generate datasets
    datasetPath = args.dataset_folder
    trainset_path = os.path.join(datasetPath, 'trainingSet.pkl')
    trainset_indice_path = os.path.join(datasetPath, 'trainingSetIndices.txt')
    if os.path.isfile(trainset_path):
        print("Loading Training Set...")
        with open(trainset_path, 'rb') as handle:
            trainingSet = pickle.load(handle)
        trainingSetIndices = np.loadtxt(trainset_indice_path, dtype='float')
        trainingSetIndices = [int(x) for x in trainingSetIndices]
        trainingSetSize = len(trainingSetIndices)*args.training_go_overs

    else:
        import sys
        sys.exit(1)

    validset_path = os.path.join(datasetPath, 'validationSet.pkl')
    validset_indice_path = os.path.join(datasetPath, 'validationSetIndices.txt')
    if os.path.isfile(validset_path):
        print("Loading Validation Set...")
        with open(validset_path, 'rb') as handle:
            validationSet = pickle.load(handle)
        validationSetIndices = np.loadtxt(validset_indice_path, dtype='float')
        validationSetIndices = [int(x) for x in validationSetIndices]
        validationSetSize = len(validationSetIndices)*args.validation_go_overs

    else:
        import sys
        sys.exit(1)

    ###################################################################################################
    # Training
    ###################################################################################################

    # Prepare network, optimizer & loss function
    model, _, _ = f.initNet('', args.learning_rate, args.momentum)

    result_logging = f.initLog_result(args.result_dir, model)

    train_labels_mean = np.loadtxt(os.path.join(args.label_folder, args.label_mean_filename), dtype='float', delimiter=',')
    train_labels_std = np.loadtxt(os.path.join(args.label_folder, args.label_std_filename), dtype='float', delimiter=',')
    
    logdir = args.log_dir

    for weight in ['best_weight', 'last_weight']:
        model_weight = os.path.join(logdir, 'weights/' + weight + '.pt')  

        model.load_state_dict(torch.load((model_weight)))

        model.eval()
        trainingBatches = f.generateBatches(trainingSet, trainingSetIndices, 1, args.training_go_overs, label)
        validationBatches = f.generateBatches(validationSet, validationSetIndices, 1, args.validation_go_overs, label)

        train_L2_mean, train_L2_std, train_L2_max, _ = f.testModel(model, train_labels_mean, train_labels_std, trainingBatches, trainingSetSize, result_logging)
        test_L2_mean, test_L2_std, test_L2_max, forward_time = f.testModel(model, train_labels_mean, train_labels_std, validationBatches, validationSetSize, result_logging)

        result_logging.info("Model weight: %s", model_weight)
        result_logging.info("Total testing forward time %f s", forward_time)
        average_forward_time = forward_time/validationSetSize
        result_logging.info("Average forward time %f s, %f Hz", average_forward_time, 1/average_forward_time)
        result_logging.info("Training Set Error:")
        result_logging.info("P1 error (mm) %f +- %f with max %f", train_L2_mean[0], train_L2_std[0], train_L2_max[0])
        result_logging.info("P1 error percentage %f +- %f with max %f", 
                            train_L2_mean[0]/335*100, 
                            train_L2_std[0]/335*100, 
                            train_L2_max[0]/335*100)
        result_logging.info("P2 error (mm) %f +- %f with max %f", train_L2_mean[1], train_L2_std[1], train_L2_max[1])
        result_logging.info("P2 error percentage %f +- %f with max %f", 
                            train_L2_mean[1]/335*100, 
                            train_L2_std[1]/335*100, 
                            train_L2_max[1]/335*100)
        result_logging.info("P3 error (mm) %f +- %f with max %f", train_L2_mean[2], train_L2_std[2], train_L2_max[2])
        result_logging.info("P3 error percentage %f +- %f with max %f", 
                            train_L2_mean[2]/335*100, 
                            train_L2_std[2]/335*100, 
                            train_L2_max[2]/335*100)
        result_logging.info("Testing Set Error:")
        result_logging.info("P1 error (mm) %f +- %f with max %f", test_L2_mean[0], test_L2_std[0], test_L2_max[0])
        result_logging.info("P1 error percentage %f +- %f with max %f", 
                            test_L2_mean[0]/335*100, 
                            test_L2_std[0]/335*100, 
                            test_L2_max[0]/335*100)
        result_logging.info("P2 error (mm) %f +- %f with max %f", test_L2_mean[1], test_L2_std[1], test_L2_max[1])
        result_logging.info("P2 error percentage %f +- %f with max %f", 
                            test_L2_mean[1]/335*100, 
                            test_L2_std[1]/335*100, 
                            test_L2_max[1]/335*100)
        result_logging.info("P3 error (mm) %f +- %f with max %f", test_L2_mean[2], test_L2_std[2], test_L2_max[2])
        result_logging.info("P3 error percentage %f +- %f with max %f", 
                            test_L2_mean[2]/335*100, 
                            test_L2_std[2]/335*100, 
                            test_L2_max[2]/335*100)


if __name__ == '__main__':
    main()
