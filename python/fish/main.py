###################################################################################################
# Imports
###################################################################################################
import os
import numpy as np
import torch
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

    # Create training and validation indices to select from dataset
    trainingSetIndices, validationSetIndices = f.defineSet()
    # testingSetSize = len(testingSetIndices)

    # Generate datasets
    datasetPath = args.dataset_folder
    trainset_path = os.path.join(datasetPath, 'trainingSet.pkl')
    trainset_indice_path = os.path.join(datasetPath, 'trainingSetIndices.txt')
    trainset_label_path = os.path.join(datasetPath, 'trainingSetLabels.txt')

    if os.path.isfile(trainset_path):
        print("Loading Training Set...")
        with open(trainset_path, 'rb') as handle:
            trainingSet = pickle.load(handle)
        trainingSetIndices = np.loadtxt(trainset_indice_path, dtype='float')
        trainingSetIndices = [int(x) for x in trainingSetIndices]
        trainingSetSize = len(trainingSetIndices)*args.training_go_overs
        trainingSetLabels = np.loadtxt(trainset_label_path, dtype='float')

        trainingSet = torch.FloatTensor(trainingSet)
        trainingSetLabels = torch.FloatTensor(trainingSetLabels).to(args.device)

    else:
        print("No Training Set Found.")
        import sys
        sys.exit(1)

    validset_path = os.path.join(datasetPath, 'validationSet.pkl')
    validset_indice_path = os.path.join(datasetPath, 'validationSetIndices.txt')
    validset_label_path = os.path.join(datasetPath, 'validationSetLabels.txt')

    if os.path.isfile(validset_path):
        print("Loading Validation Set...")
        with open(validset_path, 'rb') as handle:
            validationSet = pickle.load(handle)
        validationSetIndices = np.loadtxt(validset_indice_path, dtype='float')
        validationSetIndices = [int(x) for x in validationSetIndices]
        validationSetSize = len(validationSetIndices)*args.validation_go_overs
        validationSetLabels = np.loadtxt(validset_label_path, dtype='float')

        validationSet = torch.FloatTensor(validationSet).to(args.device)
        validationSetLabels = torch.FloatTensor(validationSetLabels).to(args.device)

    else:
        print("No Validation Set Found.")
        import sys
        sys.exit(1)

    ###################################################################################################
    # Training
    ###################################################################################################
    dirName = os.path.dirname(__file__)

    comment = f'_LR={args.learning_rate}_M={args.momentum}_DR={args.dropout}_' \
              f'BS={args.batch_size}_DSS={args.dataset_size}__img={args.image_size}_' \
              f'Net={args.network}_Loss={args.loss_type}_' \
              f'Sched=({args.scheduler_type},{args.scheduler_active},{args.scheduler_cos_epoch}' \
              f'{args.scheduler_step_size},{args.scheduler_gamma}' \
              f')_AdamW_groups'
    valid_loss = torch.FloatTensor(np.zeros((args.epochs))).to(args.device).requires_grad_(False)
    train_loss = torch.FloatTensor(np.zeros((args.epochs))).to(args.device).requires_grad_(False)

    # Initialize logger
    logdir, logging = f.initLog()

    # Prepare network, optimizer & loss function
    model, optimizer, getLoss = f.initNet(comment, args.learning_rate, args.momentum)

    from functions import count_parameters_in_MB
    logging.info('Model:  %s Parameter count: %e MB', args.network, count_parameters_in_MB(model))

    if args.scheduler_active == True:
        if args.scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
        elif args.scheduler_type == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.scheduler_cos_epoch)
    # Run through epochs with current model
    min_valid_loss = 10e4
    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        # Create new batches dataset_img, dataset_lab, batchSize
        trainingBatches = f.generateBatchesFish(trainingSet, trainingSetLabels, args.batch_size)
        validationBatches = f.generateBatchesFish(validationSet, validationSetLabels, args.batch_size)

        # Train model
        model, train_loss[epoch] = f.trainModel(model, optimizer, getLoss, trainingBatches,
                                                                    trainingSetSize, logging)
        
        logging.info('train_loss %f', train_loss[epoch])

        # Validate model after args.valid_freq epochs
        L2_mean, L2_max, valid_loss[epoch] = f.validateModel(model, getLoss,
                                                                validationBatches, validationSetSize, logging)

        logging.info('valid_loss %f', valid_loss[epoch])

        wandb.log({"epoch": epoch,
                   "train_loss": train_loss[epoch],
                   "valid_loss": valid_loss[epoch],
                   "L2_mean_P1": L2_mean[0],
                   "L2_mean_P2": L2_mean[1],
                   "L2_max_P1": L2_max[0],
                   "L2_max_P2": L2_max[1]},
                   step = epoch)
        wandb.watch(model)

        # Save network
        if valid_loss[epoch] < min_valid_loss:
            modelName = 'weights/' + 'best_weight.pt'
            torch.save(model.state_dict(), os.path.join(logdir, modelName))
            min_valid_loss = valid_loss[epoch]
            logging.info("best weights at epoch %d with valid loss %f", epoch, valid_loss[epoch])

        if args.scheduler_active == True:
            if args.scheduler_once == True:
                if epoch <= args.scheduler_step_size + 10:
                    scheduler.step()
            else:
                scheduler.step()

    modelName = 'weights/' + 'last_weight.pt'
    torch.save(model.state_dict(), os.path.join(logdir, modelName))

    wandb.finish()
    # tb.close()

    result_logging = f.initLog_result(args.result_dir, model)

    train_labels_mean = np.loadtxt(os.path.join(args.label_folder, args.label_mean_filename), dtype='float', delimiter=',')
    train_labels_std = np.loadtxt(os.path.join(args.label_folder, args.label_std_filename), dtype='float', delimiter=',')

    for weight in ['best_weight', 'last_weight']:
        model_weight = os.path.join(logdir, 'weights/' + weight + '.pt')   
        model.load_state_dict(torch.load((model_weight)))
        model.eval()

        trainingBatches = f.generateBatchesFish(trainingSet, trainingSetLabels, args.batch_size)
        validationBatches = f.generateBatchesFish(validationSet, validationSetLabels, 1)

        train_L2_mean, train_L2_std, train_L2_max, _ = f.testModel(model, train_labels_mean, train_labels_std, trainingBatches, trainingSetSize, result_logging)
        test_L2_mean, test_L2_std, test_L2_max, forward_time = f.testModel(model, train_labels_mean, train_labels_std, validationBatches, validationSetSize, result_logging)

        result_logging.info("Model weight: %s", model_weight)
        result_logging.info("Total testing forward time %f s", forward_time)
        average_forward_time = forward_time/validationSetSize
        result_logging.info("Average forward time %f s, %f Hz", average_forward_time, 1/average_forward_time)
        result_logging.info("Training Set Error:")
        result_logging.info("P1 error (mm) %f +- %f with max %f", train_L2_mean[0], train_L2_std[0], train_L2_max[0])
        result_logging.info("P1 error percentage %f +- %f with max %f", 
                            train_L2_mean[0]/115*100, 
                            train_L2_std[0]/115*100, 
                            train_L2_max[0]/115*100)
        result_logging.info("P2 error (mm) %f +- %f with max %f", train_L2_mean[1], train_L2_std[1], train_L2_max[1])
        result_logging.info("P2 error percentage %f +- %f with max %f", 
                            train_L2_mean[1]/115*100, 
                            train_L2_std[1]/115*100, 
                            train_L2_max[1]/115*100)
        result_logging.info("P3 error (mm) %f +- %f with max %f", train_L2_mean[2], train_L2_std[2], train_L2_max[2])
        result_logging.info("P3 error percentage %f +- %f with max %f", 
                            train_L2_mean[2]/115*100, 
                            train_L2_std[2]/115*100, 
                            train_L2_max[2]/115*100)
        result_logging.info("Testing Set Error:")
        result_logging.info("P1 error (mm) %f +- %f with max %f", test_L2_mean[0], test_L2_std[0], test_L2_max[0])
        result_logging.info("P1 error percentage %f +- %f with max %f", 
                            test_L2_mean[0]/115*100, 
                            test_L2_std[0]/115*100, 
                            test_L2_max[0]/115*100)
        result_logging.info("P2 error (mm) %f +- %f with max %f", test_L2_mean[1], test_L2_std[1], test_L2_max[1])
        result_logging.info("P2 error percentage %f +- %f with max %f", 
                            test_L2_mean[1]/115*100, 
                            test_L2_std[1]/115*100, 
                            test_L2_max[1]/115*100)
        result_logging.info("P3 error (mm) %f +- %f with max %f", test_L2_mean[2], test_L2_std[2], test_L2_max[2])
        result_logging.info("P3 error percentage %f +- %f with max %f", 
                            test_L2_mean[2]/115*100, 
                            test_L2_std[2]/115*100, 
                            test_L2_max[2]/115*100)


if __name__ == '__main__':
    main()
