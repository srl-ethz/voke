###################################################################################################
# Imports
###################################################################################################
import os
import numpy as np
from datetime import datetime

import torch
import progressbar

import functions as f

import pickle
import wandb
from config import args


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def main():
    ###################################################################################################
    # Import dataset
    ###################################################################################################

    # Import labels
    labelPath = os.path.join(args.label_folder, args.label_filename)
    label = np.loadtxt(labelPath, dtype="float", delimiter=",")
    label_idx = label[:, 0]
    label_idx = [int(x) for x in label_idx]
    label = torch.FloatTensor(label).to(args.device)

    # Create training and validation indices to select from dataset
    trainingSetIndices, validationSetIndices, testingSetIndices = f.defineSetSoPrA(
        label
    )
    trainingSetSize = len(trainingSetIndices) * args.training_go_overs
    validationSetSize = len(validationSetIndices) * args.validation_go_overs
    testingSetSize = len(testingSetIndices)

    # Generate datasets
    if os.path.isfile("./data/trainingSet.pkl"):
        print("Loading Training Set...")
        with open("./data/trainingSet.pkl", "rb") as handle:
            trainingSet = pickle.load(handle)
        trainingSetIndices = np.loadtxt(
            "./data/trainingSetIndices.txt", dtype="float"
        )
        trainingSetIndices = [int(x) for x in trainingSetIndices]
    else:
        print("Creating Training Set...")
        trainingSet = f.generateSetSoPrA(trainingSetIndices, label_idx, args.data)
        np.savetxt("./data/trainingSetIndices.txt", trainingSetIndices)
        with open("./data/trainingSet.pkl", "wb") as handle:
            pickle.dump(trainingSet, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.isfile("./data/validationSet.pkl"):
        print("Loading Validation Set...")
        with open("./data/validationSet.pkl", "rb") as handle:
            validationSet = pickle.load(handle)
        validationSetIndices = np.loadtxt(
            "./data/validationSetIndices.txt", dtype="float"
        )
        validationSetIndices = [int(x) for x in validationSetIndices]

    else:
        print("\nCreating Validation Set...")
        validationSet = f.generateSetSoPrA(validationSetIndices, label_idx, args.data)
        np.savetxt("./data/validationSetIndices.txt", validationSetIndices)
        with open("./data/validationSet.pkl", "wb") as handle:
            pickle.dump(validationSet, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.isfile("./data/testingSet.pkl"):
        print("Loading Testig Set...")
        with open("./data/testingSet.pkl", "rb") as handle:
            testingSet = pickle.load(handle)
        testingSetIndices = np.loadtxt(
            "./data/testingSetIndices.txt", dtype="float"
        )
        testingSetIndices = [int(x) for x in testingSetIndices]

    else:
        print("\nCreating Testing Set...")
        testingSet = f.generateSetSoPrA(testingSetIndices, label_idx, args.data)
        np.savetxt("./data/testingSetIndices.txt", testingSetIndices)
        with open("./data/testingSet.pkl", "wb") as handle:
            pickle.dump(testingSet, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ###################################################################################################
    # Training
    ###################################################################################################
    dirName = os.path.dirname(__file__)

    comment = (
        f"_LR={args.learning_rate}_M={args.momentum}_DR={args.dropout}_"
        f"BS={args.batch_size}_DSS={args.dataset_size}__img={args.image_size}_"
        f"Net=Net_short_Loss=MAE_"
        f"Sched=({args.scheduler_type},{args.scheduler_active},{args.scheduler_cos_epoch}"
        f"{args.scheduler_step_size},{args.scheduler_gamma}"
        f")_AdamW_groups"
    )
    valid_loss = (
        torch.FloatTensor(np.zeros((args.epochs))).to(args.device).requires_grad_(False)
    )
    train_loss = (
        torch.FloatTensor(np.zeros((args.epochs))).to(args.device).requires_grad_(False)
    )
    now = datetime.now()
    currentTime = now.strftime("%y-%m-%d__%H-%M-%S_")

    # Save hyperparameters
    config = {
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "momentum": args.momentum,
        "scheduler_cos_T": args.scheduler_cos_epoch,
        "dropout ": args.dropout,
        "dataset_size": args.dataset_size,
        "loss_type": 1,
    }
    # Initialize Weights & Biases
    wandb.init(project="vise_sopra", entity=args.entity, config=config)

    # Prepare network, optimizer & loss function
    model, optimizer, getLoss, tb = f.initNet(
        comment, args.learning_rate, args.momentum, args.dropout
    )
    if args.scheduler_active == True:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.scheduler_cos_epoch
        )
    # Run through epochs with current model
    bar = progressbar.ProgressBar(
        maxval=args.epochs,
        widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
    )
    bar.start()
    for epoch in range(args.epochs):
        bar.update(epoch + 1)

        # Create new batches
        trainingBatches = f.generateBatchesSoPrA(
            trainingSet,
            trainingSetIndices,
            args.batch_size,
            args.training_go_overs,
            label,
        )
        validationBatches = f.generateBatchesSoPrA(
            validationSet,
            validationSetIndices,
            args.batch_size,
            args.validation_go_overs,
            label,
        )

        # Train model
        model, train_loss[epoch] = f.trainModel(
            model, optimizer, getLoss, trainingBatches, trainingSetSize
        )

        # Validate model after args.valid_freq epochs
        if np.mod(epoch, args.valid_freq) == 0:
            L2_mean, L2_max, valid_loss[epoch] = f.validateModel(
                model, getLoss, validationBatches, validationSetSize
            )
            f.updateTensorboard(
                tb,
                epoch,
                train_loss,
                valid_loss,
                L2_mean,
                L2_max,
            )

            wandb.log(
                {
                    "train_loss": train_loss[epoch],
                    "valid_loss": valid_loss[epoch],
                    "L2_mean_P1": L2_mean[0],
                    "L2_mean_P2": L2_mean[1],
                    "L2_max_P1": L2_max[0],
                    "L2_max_P2": L2_max[1],
                }
            )
            wandb.watch(model)

            # Save network
            modelName = "models/" + currentTime + comment + ".pt"
            torch.save(model.state_dict(), os.path.join(dirName, modelName))

        if args.scheduler_active == True:
            if args.scheduler_once == True:
                if epoch <= args.scheduler_step_size + 10:
                    scheduler.step()
            else:
                scheduler.step()

    wandb.finish()
    tb.close()


if __name__ == "__main__":
    main()
