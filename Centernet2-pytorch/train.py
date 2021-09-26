import torch
import configs as cfg


def main():
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # dataset
    from dataset.ddr_dataset import MyDataSet
    root_dir = 'D:/Datasets'
    train_dataset = MyDataSet(root_dir, 'train')
    valid_dataset = MyDataSet(root_dir, 'valid')

    # dataloader
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.NW,
        collate_fn=train_dataset.collate_fn
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.NW,
        collate_fn=valid_dataset.collate_fn
    )

    # create model
    from create_models import create_centernet2_baseline
    model = create_centernet2_baseline(cfg.NUM_CLASSES + 1)
    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.33)

    # begin trainning
    from train_utils.train_eval_utils import train_one_epoch, evaluate

    train_loss = []
    learning_rate = []
    val_map = []

    for epoch in range(cfg.START_EPOCH, cfg.END_EPOCH):
        # train for one epoch, printing every 10 iterations
        mean_loss, lr = train_one_epoch(
            model, optimizer, train_dataloader,
            device=device, epoch=epoch,
            print_freq=50, warmup=True
        )
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        coco_info = evaluate(model, valid_dataloader, device=device)

        # write into txt
        import datetime
        results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [str(round(i, 4)) for i in coco_info + [mean_loss.item()]] + [str(round(lr, 6))]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # pascal mAP

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        torch.save(save_files, "./save_weights/resNetFpn-model-{}.pth".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from train_utils.plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from train_utils.plot_curve import plot_map
        plot_map(val_map)


if __name__ == '__main__':
    main()