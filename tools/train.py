import torch
import os.path as osp
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from simdet.config import Config


def main(args):
    cfg = Config(args.config_path, args.resume_from)

    if torch.cuda.is_available():
        device = 'cuda'
        torch.backends.cudnn.benckmark = True
    else:
        device = 'cpu'

    train_dl = cfg.build_dataloader('train')
    val_dl = cfg.build_dataloader('val')
    model = cfg.build_model().to(device)
    optimizer = cfg.build_optimizer(model)
    scheduler = cfg.build_scheduler(optimizer)
    metric = cfg.build_metric()

    writer = SummaryWriter(args.out_dir)

    for e in range(cfg.start_epoch, cfg.epochs+1):
        train_loss = train_count = val_loss = val_count = 0

        model.train()
        for images, targets, _ in tqdm(train_dl, desc=f'[{e}] train'):
            images, targets = images.to(device), [target.to(device) for target in targets]
            outputs = model(images)
            loss = model.loss(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss * images.size(0)
            train_count += images.size(0)
        train_loss = (train_loss / train_count).item()
        scheduler.step()

        model.eval()
        evaluate = (e % 10 == 0) or (e == cfg.epochs)
        with torch.no_grad():
            for images, targets, metas in tqdm(val_dl, desc=f'[{e}] val'):
                images, targets = images.to(device), [target.to(device) for target in targets]
                outputs = model(images)
                loss = model.loss(outputs, targets)
                if evaluate:
                    preds = model.predict(outputs)
                    metric.update(preds, metas)
                val_loss += loss * images.size(0)
                val_count += images.size(0)
            val_loss = (val_loss / val_count).item()
            if evaluate:
                val_result = metric.compute()
                metric.reset()

        writer.add_scalar('Loss/train', train_loss, e)
        writer.add_scalar('Loss/val', val_loss, e)
        for i, last_lr in enumerate(scheduler.get_last_lr()):
            writer.add_scalar(f'LearningRate/lr_{i}', last_lr, e)
        if evaluate:
            for metric_name, val in val_result.items():
                writer.add_scalar(f'Metric/{metric_name}', val, e)

        states = {
            'epoch': e,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(states, osp.join(args.out_dir, 'latest.pth'))

        print(f'[{e}] loss: {train_loss:.04f}, val_loss: {val_loss:.04f}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('--out_dir', type=str, default='./results')
    parser.add_argument('--resume_from', type=str, default=None)
    args = parser.parse_args()

    main(args)
