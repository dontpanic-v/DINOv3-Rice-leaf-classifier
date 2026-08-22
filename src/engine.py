import math
import sys
import torch

import src.utils as utils


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
    epoch,
    print_freq=10,
    max_norm=0,
    writer=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter=" : ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    for i, (images, labels) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        images = images.to(device)
        labels = labels.to(device)

        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            if writer is not None:
                writer.flush()
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            metric_logger.update(grad_norm=grad_norm.item())
        optimizer.step()

        acc = (logits.argmax(dim=1) == labels).float().mean().item()
        metric_logger.update(loss=loss_value)
        metric_logger.update(accuracy=acc)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer is not None:
            global_step = epoch * len(data_loader) + i
            writer.add_scalar("train/loss", loss_value, global_step)
            writer.add_scalar("train/accuracy", acc, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            if max_norm > 0:
                writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)

    print("averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(model, data_loader, criterion, device, print_freq=10):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter=" : ")
    header = "Validation:"

    for images, labels in metric_logger.log_every(data_loader, print_freq, header):
        images = images.to(device)
        labels = labels.to(device)

        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        acc = (logits.argmax(dim=1) == labels).float().mean().item()
        metric_logger.update(loss=loss.item())
        metric_logger.update(accuracy=acc)

    print("averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
