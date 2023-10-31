import os
import random

import cv2
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import wandb

from config import Config
from CSNet.image_utils.image_preprocess import get_cropping_image, get_zooming_image, get_shifted_image, get_rotated_image
from dataset import BCDataset, UnlabledDataset
from test import test_while_training
from vapnet import VAPNet

Image.MAX_IMAGE_PIXELS = None

def not_convert_to_tesnor(batch):
        return batch

def build_dataloader(cfg):
    bc_dataset = BCDataset('train', cfg)
    labeled_loader = DataLoader(dataset=bc_dataset,
                              batch_size=cfg.batch_size,
                              collate_fn=not_convert_to_tesnor,
                              shuffle=True,
                              num_workers=cfg.num_workers)
    unlabeled_dataset = UnlabledDataset('train', cfg)
    unlabeled_loader = DataLoader(dataset=unlabeled_dataset,
                              batch_size=cfg.batch_size,
                              collate_fn=not_convert_to_tesnor,
                              shuffle=True,
                              num_workers=cfg.num_workers)
    return labeled_loader, unlabeled_loader

class Trainer(object):
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.model = model

        self.image_dir = self.cfg.image_dir

        self.device = torch.device('cuda:{}'.format(self.cfg.gpu_id))
        # self.device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
        self.batch_size = self.cfg.batch_size

        self.bc_loader, self.unlabeled_loader = build_dataloader(self.cfg)

        self.magnitude_loss_fn = torch.nn.L1Loss(reduction='mean')
        self.adjustment_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.suggestion_loss_fn = torch.nn.BCELoss(reduction='mean')

        self.optimizer = optim.Adam(params=model.parameters(),
                                    lr=self.cfg.learning_rate,
                                    weight_decay=self.cfg.weight_decay)
        
        self.epoch = 0
        self.max_epoch = self.cfg.max_epoch

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.mean, std=self.cfg.std)
        ])

        self.train_iter = 0
        self.suggested_case_iter = 0
        self.magnitude_loss_sum = 0
        self.adjustment_loss_sum = 0
        self.suggestion_loss_sum = 0

    def training(self):
        print('\n======train start======\n')
        self.model.train().to(self.device)
        bc_iter = iter(self.bc_loader)
        for index, data in enumerate(self.unlabeled_loader):

            self.train_iter += 1
            try:
                bc_data_list = next(bc_iter)
            except:
                bc_iter = iter(self.bc_loader)
            unlabeled_data_list = data

            # get randomly perturbed or not perturbed image and label
            l_image_list, l_magnitude_label_list, l_adjustment_label_list, l_suggestion_label_list = self.get_labeled_data_list(bc_data_list)

            # get unlabeled data label
            ul_image_list = [Image.open(os.path.join('../VAPNet/data/open_images', x[0])).convert('RGB') for x in unlabeled_data_list]
            ul_magnitude_label_list = [x[1] for x in unlabeled_data_list]
            ul_adjustment_label_list = [x[2] for x in unlabeled_data_list]
            ul_suggestion_label_list = [x[3] for x in unlabeled_data_list]

            # combine
            image_list = l_image_list + ul_image_list
            gt_suggestion_list = l_suggestion_label_list + ul_suggestion_label_list
            gt_magnitude_list = l_magnitude_label_list + ul_magnitude_label_list
            gt_adjustment_list = l_adjustment_label_list + ul_adjustment_label_list

            # shuffle
            combined_list = list(zip(image_list, gt_magnitude_list, gt_adjustment_list, gt_suggestion_list))
            random.shuffle(combined_list)
            image_list, gt_magnitude_list, gt_adjustment_list, gt_suggestion_list = [list(x) for x in zip(*combined_list)]
            
            # model inference
            predicted_suggestion, predicted_adjustment, predicted_magnitude  = self.model(self.convert_image_list_to_tensor(image_list).to(self.device))

            selected_gt_adjustment_list = []
            selected_predicted_adjustment = []
            selected_gt_magnitude_list = []
            selected_predicted_magnitude = []
            for index, suggestion_label in enumerate(gt_suggestion_list):
                if suggestion_label == [0.0]:
                    continue
                selected_gt_magnitude_list.append(gt_magnitude_list[index])
                selected_predicted_magnitude.append(predicted_magnitude[index])
                selected_gt_adjustment_list.append(gt_adjustment_list[index])
                selected_predicted_adjustment.append(predicted_adjustment[index])
            
            gt_magnitude_list = selected_gt_magnitude_list
            predicted_magnitude = torch.stack(selected_predicted_magnitude)
            gt_adjustment_list = selected_gt_adjustment_list
            predicted_adjustment = torch.stack(selected_predicted_adjustment)

            gt_magnitude_list = torch.tensor(gt_magnitude_list).to(self.device)
            gt_adjustment_list = torch.tensor(gt_adjustment_list).to(self.device)

            # calculate suggestion, adjustment, magnitude loss using BCELoss, L1 Loss
            suggestion_loss = self.suggestion_loss_fn(predicted_suggestion, gt_suggestion_list)
            magnitude_loss = self.magnitude_loss_fn(predicted_magnitude, gt_magnitude_list)
            adjustment_loss = self.adjustment_loss_fn(predicted_adjustment, gt_adjustment_list)

            total_loss = suggestion_loss + adjustment_loss + magnitude_loss
            train_log = f'suggestion loss: {suggestion_loss.item():.5f}/adjustment loss: {adjustment_loss.item():.5f}/magnitude loss:{magnitude_loss.item():.5f}'
            print(train_log)

            self.suggestion_loss_sum += suggestion_loss.item()
            self.magnitude_loss_sum += magnitude_loss.item()
            self.adjustment_loss_sum += adjustment_loss.item()

            # back propagation
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            self.suggested_case_iter += 1

            if self.train_iter % 20 == 0:
                ave_suggestion_loss = self.suggestion_loss_sum / self.suggested_case_iter
                ave_magnitude_loss = self.magnitude_loss_sum / self.suggested_case_iter
                ave_adjustment_loss = self.adjustment_loss_sum / self.suggested_case_iter

                wandb.log({"Train Loss/suggestion_loss": ave_suggestion_loss, "Train Loss/magnitude_loss": ave_magnitude_loss, "Train Loss/adjustment_loss": ave_adjustment_loss})
                self.magnitude_loss_sum = 0
                self.suggested_case_iter = 0
                self.adjustment_loss_sum = 0
                self.suggestion_loss_sum = 0

            """
            if self.train_iter % 5000 == 0:
                checkpoint_path = os.path.join(self.cfg.weight_dir, 'checkpoint-weight.pth')
                torch.save(self.model.state_dict(), checkpoint_path)
                print('Checkpoint Saved...\n')
                
                for i in range(0.5, 1.0, 0.1):
                    test_while_training(threshold=i)
                test_while_training()
            """
            
        print('\n======train end======\n')

    def convert_image_list_to_tensor(self, image_list):
        tensor = []
        for image in image_list:
            # Grayscale to RGB
            if len(image.getbands()) == 1:
                rgb_image = Image.new("RGB", image.size)
                rgb_image.paste(image, (0, 0, image.width, image.height))
                image = rgb_image
            np_image = np.array(image)
            np_image = cv2.resize(np_image, self.cfg.image_size)
            tensor.append(self.transformer(np_image))
        tensor = torch.stack(tensor, dim=0)
        
        return tensor

    def run(self):
        for epoch in range(self.epoch, self.max_epoch):
            self.epoch = epoch + 1
            self.training()

            # save checkpoint
            checkpoint_path = os.path.join(self.cfg.weight_dir, 'checkpoint-weight.pth')
            torch.save(self.model.state_dict(), checkpoint_path)
            print('Checkpoint Saved...\n')

            
            test_while_training(adjustment_threshold=self.cfg.adjustment_threshold)

            epoch_log = 'epoch: %d / %d, lr: %8f' % (self.epoch, self.max_epoch, self.optimizer.param_groups[0]['lr'])
            print(epoch_log)

            self.magnitude_loss_sum = 0
            self.adjustment_loss_sum = 0
            self.suggestion_loss_sum = 0
            self.train_iter = 0
            self.suggested_case_iter = 0

    def get_perturbed_image(self, data):
        image_name = data[0]
        image = Image.open(os.path.join(self.image_dir, image_name))
        best_crop_bounding_box = data[1]

        func_choice = random.randint(0, 1)
        if func_choice == 0:
            return image.crop(best_crop_bounding_box), (0.0, 0.0), [0.0, 0.0, 0.0, 0.0], [0.0]
        elif func_choice == 1:
            output = get_shifted_image(image, best_crop_bounding_box, allow_zero_pixel=False, option='vapnet')
        """
        elif func_choice == 2 :
            output = get_rotated_image(image, best_crop_bounding_box, allow_zero_pixel=False, option='vapnet')
        """
        if output == None:
            return None
        perturbed_image, operator = output
        
        adjustment_label = [0.0] * 4
        if operator[0] < 0:
            operator[0] = abs(operator[0])
            adjustment_label[1] = 1.0
        elif operator[1] > 0:
            adjustment_label[0] = 1.0
        
        if operator[1] < 0:
            operator[1] = abs(operator[1])
            adjustment_label[3] = 1.0
        elif operator[1] > 0:
            adjustment_label[2] = 1.0
    
        return perturbed_image, (operator[0], operator[1]), adjustment_label, [1.0]

    def get_labeled_data_list(self, bc_data_list):
        image_list = []
        magnitude_label_list = []
        adjustment_label_list = []
        suggestion_label_list = []

        for data in bc_data_list:
            labeled_data = self.get_perturbed_image(data)
            if labeled_data == None:
                continue
            image_list.append(labeled_data[0])
            magnitude_label_list.append(labeled_data[1])
            adjustment_label_list.append(labeled_data[2])
            suggestion_label_list.append(labeled_data[3])

        return image_list, magnitude_label_list, adjustment_label_list, suggestion_label_list

if __name__ == '__main__':
    cfg = Config()
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="vapnet.v4",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": cfg.learning_rate,
        "architecture": "CNN",
        "dataset": "BC/UN",
        "epochs": cfg.max_epoch,
        "memo": "None"
        }
    )
    
    model = VAPNet(cfg)
    # weight_file = os.path.join(cfg.weight_dir, 'checkpoint-weight.pth')
    # model.load_state_dict(torch.load(weight_file))

    trainer = Trainer(model, cfg)
    trainer.run()

    wandb.finish()