import os

import cv2
from shapely.geometry import Polygon
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score as f1
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import wandb

from config import Config
from dataset import LabledDataset
from CSNet.image_utils.image_preprocess import get_shifted_box, get_rotated_box
from vapnet import VAPNet

def build_dataloader(cfg):
    labeled_dataset = LabledDataset('test', cfg)
    data_loader = DataLoader(dataset=labeled_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=False,
                              num_workers=cfg.num_workers)
    return data_loader

class Tester(object):
    def __init__(self, model, cfg, adjustment_threeshold):
        self.cfg = cfg
        self.model = model

        self.image_dir = os.path.join(self.cfg.image_dir, 'image_labeled_vapnet')

        self.data_loader = build_dataloader(self.cfg)
        self.device = torch.device('cuda:{}'.format(self.cfg.gpu_id))
        # self.device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
        # self.device = torch.device('cpu')

        self.batch_size = self.cfg.batch_size

        self.adjustment_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.magnitude_loss_fn = torch.nn.L1Loss(reduction='mean')

        self.adjustment_threshold = np.array(adjustment_threeshold)
        self.adjustment_count = self.cfg.adjustment_count

        self.data_length = self.data_loader.__len__()

        self.magnitude_loss_sum = 0
        self.adjustment_loss_sum = 0
        self.iou_score_sum = 0

    def run(self, custom_threshold=0):
        print('\n======test start======\n')

        total_gt_magnitude_label = np.array([])
        total_predicted_magnitude = np.array([])
        total_gt_adjustment_label = np.array([])
        total_one_hot_predicted_adjustment = np.array([])

        total_gt_perturbed_bounding_box = []
        total_gt_bounding_box = []
        total_image_size = []

        self.model.eval().to(self.device)
        with torch.no_grad():
            for index, data in tqdm(enumerate(self.data_loader), total=self.data_length):
                # data split
                image = data[0].to(self.device)
                image_size = data[1].tolist()
                gt_bounding_box = data[2].tolist()
                gt_perturbed_bounding_box = data[3].tolist()
                gt_magnitude_label = data[4].to(self.device)
                gt_adjustment_label = data[5].to(self.device)

                # model inference
                predicted_magnitude, predicted_adjustment = self.model(image.to(self.device))

                selected_gt_magnitude_label = []
                selected_predicted_magnitude = []

                for index, adjustment_label in enumerate(predicted_adjustment):
                    if adjustment_label[4] >= self.adjustment_threshold[4]:
                        continue
                    selected_gt_magnitude_label.append(gt_magnitude_label[index])
                    selected_predicted_magnitude.append(predicted_magnitude[index])

                selected_gt_magnitude_label = torch.stack(selected_gt_magnitude_label)
                selected_predicted_magnitude = torch.stack(selected_predicted_magnitude)

                # caculate loss
                self.magnitude_loss_sum += self.magnitude_loss_fn(selected_predicted_magnitude, selected_gt_magnitude_label)
                self.adjustment_loss_sum += self.adjustment_loss_fn(predicted_adjustment, gt_adjustment_label)

                # convert tensor to numpy for using sklearn metrics
                gt_magnitude_label = gt_magnitude_label.to('cpu').numpy()
                predicted_magnitude = predicted_magnitude.to('cpu').numpy()
                gt_adjustment_label = gt_adjustment_label.to('cpu').numpy()
                predicted_adjustment = predicted_adjustment.to('cpu').numpy()

                one_hot_predicted_adjustment = np.apply_along_axis(self.convert_array_to_one_hot_encoded, arr=predicted_adjustment, axis=1)

                total_gt_magnitude_label = self.add_to_total(gt_magnitude_label, total_gt_magnitude_label)
                total_predicted_magnitude = self.add_to_total(predicted_magnitude, total_predicted_magnitude)
                total_gt_adjustment_label = self.add_to_total(gt_adjustment_label, total_gt_adjustment_label)
                total_one_hot_predicted_adjustment = self.add_to_total(one_hot_predicted_adjustment, total_one_hot_predicted_adjustment)
                
                total_gt_bounding_box += gt_bounding_box
                total_gt_perturbed_bounding_box += gt_perturbed_bounding_box
                total_image_size += image_size
        
        f1_score = self.calculate_f1_score(total_gt_adjustment_label, total_one_hot_predicted_adjustment)

        # get predicted bounding box
        predicted_bounding_box = []
        
        for index, gt_perturbed_box in enumerate(total_gt_perturbed_bounding_box):
            
            adjustment_label = total_one_hot_predicted_adjustment[index]
            if adjustment_label[4] == 1.0:
                magnitude = [0.0, 0.0]
            else:
                magnitude = total_predicted_magnitude[index].copy()

                if adjustment_label[0] == 0.0 and adjustment_label[1] == 0.0:
                    magnitude[0] = 0.0
                elif adjustment_label[0] == 1:
                    magnitude[0] = -magnitude[0]
                elif adjustment_label[1] == 1:
                    magnitude[0] = magnitude[0]

                if adjustment_label[2] == 0.0 and adjustment_label[3] == 0.0:
                    magnitude[1] = 0.0
                elif adjustment_label[2] == 1:
                    magnitude[1] = -magnitude[1]
                elif adjustment_label[3] == 1:
                    magnitude[1] = magnitude[1]

            predicted_box = get_shifted_box(image_size=total_image_size[index], \
                                            bounding_box_corners=gt_perturbed_box, \
                                            mag=magnitude)
            predicted_bounding_box.append(predicted_box)
        # calculate average iou score for each bounding box pairs
        iou_score = self.calculate_ave_iou_score(total_gt_bounding_box, predicted_bounding_box)

        print('\n======test end======\n')

        ave_magnitude_loss = self.magnitude_loss_sum  / self.data_length
        ave_adjustment_loss = self.adjustment_loss_sum / self.data_length

        loss_log = f'{ave_adjustment_loss}/{ave_magnitude_loss}'
        accuracy_log = f'{f1_score}/{iou_score:.5f}'
    
        print(loss_log)
        print(accuracy_log)
        
        wandb.log({"Test/test_magnitude_loss": ave_magnitude_loss, "Test/test_adjustment_loss": ave_adjustment_loss})
        wandb.log({
            f"Test/f1-score(left)": f1_score[0],
            f"Test/f1-score(right)": f1_score[1],
            f"Test/f1-score(up)": f1_score[2],
            f"Test/f1-score(down)": f1_score[3],
            f"Test/f1-score(no-suggestion)": f1_score[4],
            f"Test/IoU": iou_score
        })
    
    def add_to_total(self, target_np_array, total_np_array):
        if total_np_array.shape == (0,):
            total_np_array = target_np_array
        else:
            total_np_array = np.concatenate((total_np_array, target_np_array))
        return total_np_array
    
    def convert_array_to_one_hot_encoded(self, array):
        sigmoid_array = torch.sigmoid(torch.tensor(array)).numpy()
        if sigmoid_array[4] >= self.adjustment_threshold[4]:
            sigmoid_array[:4] = 0.0
            sigmoid_array[4] = 1.0
            one_hot_encoded = sigmoid_array
        else:
            sigmoid_array[4] = 0.0
            boolean_encoded = sigmoid_array >=  self.adjustment_threshold
            one_hot_encoded = boolean_encoded.astype(float)
            if one_hot_encoded[0] == 1.0 and one_hot_encoded[1] == 1.0:
                with open('exception.csv', 'a') as f:
                    f.writelines(f'{array},{sigmoid_array}\n')
                one_hot_encoded[0] = 0.0
                one_hot_encoded[1] = 0.0
            if one_hot_encoded[2] == 1.0 and one_hot_encoded[3] == 1.0:
                with open('exception.csv', 'a') as f:
                    f.writelines(f'{array},{sigmoid_array}\n')
                one_hot_encoded[2] = 0.0
                one_hot_encoded[3] = 0.0
                
        return one_hot_encoded
    
    def calculate_f1_score(self, gt_adjustment, one_hot_encoded_predicted_adjustment):
        
        if len(gt_adjustment) == 0:
            return [0.0] * self.adjustment_count

        f1_score = f1(gt_adjustment, one_hot_encoded_predicted_adjustment, average=None, zero_division=0.0)

        return f1_score

    def calculate_ave_iou_score(self, boudning_box_list, perturbed_box_list):
        # box format: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] (counter-clockwise order)
        def calculate_iou_score(box1, box2):
            # print("gt_box:", box1, "/predicted_box:", box2)
            poly1 = Polygon(box1)
            poly2 = Polygon(box2)
            if poly1.intersects(poly2) == False:
                return 0
            intersection_area = poly1.intersection(poly2).area
            union_area = poly1.union(poly2).area
            # print(intersection_area, union_area)

            iou = intersection_area / union_area if union_area > 0 else 0.0
            return iou
        
        iou_sum = 0
        for i in range(len(boudning_box_list)):
            iou_sum += calculate_iou_score(boudning_box_list[i], perturbed_box_list[i])
        
        ave_iou = iou_sum / len(boudning_box_list)
        return ave_iou

def test_while_training(adjustment_threshold):
    cfg = Config()

    model = VAPNet(cfg)
    weight_file = os.path.join(cfg.weight_dir, 'checkpoint-weight.pth')
    model.load_state_dict(torch.load(weight_file))

    tester = Tester(model, cfg, adjustment_threshold)
    tester.run()

if __name__ == '__main__':
    cfg = Config()

    model = VAPNet(cfg)
    weight_file = os.path.join(cfg.weight_dir, 'checkpoint-weight.pth')
    model.load_state_dict(torch.load(weight_file))

    tester = Tester(model, cfg)
    tester.run()