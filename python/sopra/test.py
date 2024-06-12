###################################################################################################
# Imports
###################################################################################################
import os
import numpy as np
import torch
import progressbar
import functions as f
import pickle
import wandb
from config import args
import random

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def main():
    ###################################################################################################
    # Import dataset
    ###################################################################################################


    # Save hyperparameters
    config = {
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "momentum": args.momentum,
        "scheduler_cos_epoch": args.scheduler_cos_epoch,
        "scheduler": "StepLR_" + str(args.scheduler_gamma),
        "dropout ": args.dropout,
        "dataset_size": args.dataset_size,
        "loss_type": args.loss_type,
        "network": args.network
    }
    # Initialize Weights & Biases
    wandb_dir = '/cluster/scratch/zhengh'
    # f.make_dir(wandb_dir)
    wandb.init(dir=wandb_dir, project=args.wnb_project, group=args.network, name=args.wnb_name, entity="hehuizheng", config=config)

    # Import labels
    labelPath = os.path.join(args.label_folder, args.label_filename)
    label = np.loadtxt(labelPath, dtype='float', delimiter=',')
    label_idx = label[:, 0]
    label_idx = [int(x) for x in label_idx]
    # label = label[:, 1:7]
    label = torch.FloatTensor(label).to(args.device)

    # Create training and validation indices to select from dataset
    trainingSetIndices, validationSetIndices, testingSetIndices = f.defineSetSoPrA(label)

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
        if args.train_occlusion:
            print("Augmenting Training Set...")
            trainingSet_occlusion = trainingSet.copy()
            trainingSetIndices = trainingSetIndices + trainingSetIndices
            bar_dataaug = progressbar.ProgressBar(maxval=len(trainingSetIndices),
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar_dataaug.start()
            for i in range(len(trainingSetIndices)):
                bar_dataaug.update(i)
                occlusion_width = random.randint(5, 20)
                pos = random.randint(0, args.image_size - occlusion_width)
                trainingSet_occlusion[i, :, pos:pos + occlusion_width, :] = 0
            trainingSet = np.concatenate([trainingSet, trainingSet_occlusion], axis=0)

    else:
        print("Creating Training Set...")
        trainingSet = f.generateSetSoPrA(trainingSetIndices, label_idx, args.data)
        np.savetxt(trainset_indice_path, trainingSetIndices)
        trainingSetSize = len(trainingSetIndices)*args.training_go_overs
        with open(trainset_path, 'wb') as handle:
            pickle.dump(trainingSet, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
        print("\nCreating Validation Set...")
        validationSet = f.generateSetSoPrA(validationSetIndices, label_idx, args.data)
        np.savetxt(validset_indice_path, validationSetIndices)
        validationSetSize = len(validationSetIndices)*args.validation_go_overs
        with open(validset_path, 'wb') as handle:
            pickle.dump(validationSet, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ###################################################################################################
    # Testing
    ###################################################################################################

    # Prepare network, optimizer & loss function
    model, optimizer, getLoss = f.initNet('', args.learning_rate, args.momentum)


    result_logging = f.initLog_result(args.result_dir, model)

    train_labels_mean = np.loadtxt(os.path.join(args.label_folder, args.label_mean_filename), dtype='float', delimiter=',')
    train_labels_std = np.loadtxt(os.path.join(args.label_folder, args.label_std_filename), dtype='float', delimiter=',')
    logdir = args.log_dir

    for weight in ['best_weight', 'last_weight']:
        model_weight = os.path.join(logdir, 'weights/' + weight + '.pt')   
        
        # model.load_state_dict(torch.load((model_weight)))

        state_dict =  torch.load((model_weight))
        for key in list(state_dict.keys()):
            state_dict['model.'+key] = state_dict.pop(key)
        model.load_state_dict(state_dict)
        
        model.eval()
        trainingBatches = f.generateBatchesSoPrA(trainingSet, trainingSetIndices, 1, args.training_go_overs, label)
        validationBatches = f.generateBatchesSoPrA(validationSet, validationSetIndices, 1, args.validation_go_overs, label)

        train_L2_mean, train_L2_std, train_L2_max, _ = f.testModel(model, train_labels_mean, train_labels_std, trainingBatches, trainingSetSize, result_logging)
        test_L2_mean, test_L2_std, test_L2_max, forward_time = f.testModel(model, train_labels_mean, train_labels_std, validationBatches, validationSetSize, result_logging)

        result_logging.info("Model weight: %s", model_weight)
        result_logging.info("Total testing forward time %f s", forward_time)
        average_forward_time = forward_time/validationSetSize
        result_logging.info("Average forward time %f s, %f Hz", average_forward_time, 1/average_forward_time)
        result_logging.info("Training Set Error:")
        result_logging.info("P1 error (mm) %f +- %f with max %f", train_L2_mean[0]*1000, train_L2_std[0]*1000, train_L2_max[0]*1000)
        result_logging.info("P1 error percentage %f +- %f with max %f", 
                            train_L2_mean[0]/0.270*100, 
                            train_L2_std[0]/0.270*100, 
                            train_L2_max[0]/0.270*100)
        result_logging.info("P2 error (mm) %f +- %f with max %f", train_L2_mean[1]*1000, train_L2_std[1]*1000, train_L2_max[1]*1000)
        result_logging.info("P2 error percentage %f +- %f with max %f", 
                            train_L2_mean[1]/0.270*100, 
                            train_L2_std[1]/0.270*100, 
                            train_L2_max[1]/0.270*100)
        result_logging.info("Testing Set Error:")
        result_logging.info("P1 error (mm) %f +- %f with max %f", test_L2_mean[0]*1000, test_L2_std[0]*1000, test_L2_max[0]*1000)
        result_logging.info("P1 error percentage %f +- %f with max %f", 
                            test_L2_mean[0]/0.270*100, 
                            test_L2_std[0]/0.270*100, 
                            test_L2_max[0]/0.270*100)
        result_logging.info("P2 error (mm) %f +- %f with max %f", test_L2_mean[1]*1000, test_L2_std[1]*1000, test_L2_max[1]*1000)
        result_logging.info("P2 error percentage %f +- %f with max %f", 
                            test_L2_mean[1]/0.270*100, 
                            test_L2_std[1]/0.270*100, 
                            test_L2_max[1]/0.270*100)
    


if __name__ == '__main__':
    main()
