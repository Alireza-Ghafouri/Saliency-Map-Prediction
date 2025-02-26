import torch
import numpy as np
import time
import copy
import os
from tqdm import tqdm


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    dataset_sizes,
    device,
    save_path,
    num_epochs,
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    train_epoch_losses = []
    test_epoch_losses = []
    epochs_lr_list = []

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            if phase == "train":
                scheduler.step()
                print("LR:{}".format(optimizer.param_groups[0]["lr"]))

            epoch_loss = running_loss / dataset_sizes[phase]

            print("{} Loss: {:.4f}".format(phase, epoch_loss))

            if phase == "train":
                train_epoch_losses.append(epoch_loss)
            elif phase == "val":
                test_epoch_losses.append(epoch_loss)

            epochs_lr_list.append(optimizer.param_groups[0]["lr"])

            # deep copy the model
            if phase == "val" and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

        if epoch % 5 == 4:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                },
                os.path.join(save_path, f"model_{epoch+1}.pt"),
            )

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Loss: {:4f}".format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_epoch_losses, test_epoch_losses
