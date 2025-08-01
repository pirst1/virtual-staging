import sys
import os
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
CON_DIR = os.path.abspath(os.path.join(CUR_DIR, "..", "ControlNet"))
if CON_DIR not in sys.path:
    sys.path.insert(0, CON_DIR)

from constants import AGNOSTIC, STAGED, EMPTY, CAPTION_STAGED
from share import *
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from dataset import PairedDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import argparse


def main(args):
    # Configs
    # We initialize with the pretrained ControlNet checkpoint s.t. the
    # model already knows to use the "RGB" information in the control input.
    resume_path = './models/control_sd15_pretrained_inpainter.ckpt'
    batch_size = 4
    logger_freq = 100
    learning_rate = 5e-6
    sd_locked = True
    only_mid_control = False

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(
        './models/control_v11p_sd15_inpaint.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Misc
    if args.mode == "masked_to_staged":
        dataset = PairedDataset(source_key=AGNOSTIC,
                                target_key=STAGED, caption_key=CAPTION_STAGED)
    elif args.mode == "empty_to_staged":
        dataset = PairedDataset(
            source_key=EMPTY, target_key=STAGED, caption_key=CAPTION_STAGED)

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)

    # Train
    trainer = pl.Trainer(enable_checkpointing=True, gpus=1, max_epochs=2,
                         precision=32, callbacks=[logger])
    trainer.fit(model, dataloader)
    
    torch.save(model.state_dict(), f'./models/{args.mode}.ckpt')
    print('Done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str,
                        choices=["masked_to_staged", "empty_to_staged"])
    args = parser.parse_args()
    main(args)
