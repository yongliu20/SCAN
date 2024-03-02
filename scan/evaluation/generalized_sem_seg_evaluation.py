# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import itertools
import json
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from detectron2.evaluation import SemSegEvaluator
import copy

# ADE-150
# father_class = {25: 1, 26: 21, 9: 17, 11: 6, 30: 31, 19: 31, 35: 10, 45: 10, 48: 1, 52: 6, 54: 6, 56: 15, 33: 15, 59: 121, 53: 121, 60: 21, 62: 55, 63: 58, 64: 15, 66: 17, 68: 16, 69: 19, 75: 31, 79: 25, 91: 6, 85: 82, 36: 82, 87: 82, 99: 45, 110: 19, 121: 59, 128: 21, 113: 21}

# ADE-847
father_class = {8:4, 12:16, 22:24, 29:[18, 34], 18:34, 35:24, 36:17, 41:4, 46:10, 52:4, 59:1, 63:3, 64:20, 65:[17,36], 68:10, 71:24, 73:16, 74: 28, 77:[43,106], 43:[77, 106], 78:[32, 82], 81:[34,18], 82:32, 32:82, 83:[18,34], 89:14, 90:15, 96:1, 99:125, 100:10, 101:26, 103:108, 105:[82, 32], 106:[43, 77], 111:24, 113:38, 115:[82, 32], 118:9, 123:107, 107:123, 133:[34,18], 134:24, 136:24, 143:[4,9,118], 150:27, 27:150, 153:147, 147:153, 154:141, 141:154, 160:61, 161:155, 163:33, 165:104, 172:10, 180:37, 181:94, 189:50, 191:114, 195:[3,16], 63:16, 196:20, 197:79, 199:142, 142:199, 202:102, 206:[32,82], 198:[208,421, 768], 208:421, 213:54, 216:251, 228:96, 231:13, 234:152, 236:173, 149:[173,510], 145:173, 120:173, 239:[191,114], 243:13, 246:232, 252:114, 253:178, 259:104, 261:23, 171:268, 117:271, 271:117, 275:173, 278:16, 280:235, 235:280, 283:102, 284:17, 292:247, 293:220, 295:57, 297:[34,18], 301:104, 306:37, 310:260, 311:152, 312:20, 314:102, 316:33, 319:[34,18], 320:104, 325:10, 330:[34,18], 337:286, 338:54, 679:344, 345:173, 348:251, 349:17, 350:102, 353:73, 356:61, 358:16, 167:[362,413], 363:152, 375:152, 376:7, 7:376, 378:10, 379:178, 380:70, 383:33, 384:357, 385:142, 142:385, 387:[3,496], 388:96, 389:[421,768, 198], 394:[37, 306], 397:173, 399:339, 339:399, 400:477, 404:[21, 7, 376], 412:214, 214:412, 414:[400,477], 417:[12, 16], 419:152, 424:28, 355:426, 426:355, 428:498, 432:[413,822], 436:173, 438:201, 439:4, 442:315, 443:17, 445:33, 449:[17,36], 458:20, 463:325, 325:463, 471:119, 473:24, 476:96, 477:400, 481:102, 482:4, 483:365, 365:483, 484:102, 487:444, 489:28, 494:15, 496:387, 498:428, 500:104, 501:95, 639:95, 508:104, 510:[149,173], 515:114, 517:102, 520:28, 527:[17, 36], 529:801, 131:801, 531:28, 535:234, 536:136, 538:18, 540:178, 544:232, 546:24, 547:152, 551:[18, 36], 552:50, 556:212, 561:28, 569:104, 572:61, 575:271, 554:578, 578:554, 580:286, 286:580, 581:14, 582:4, 587:154, 588:17, 17:588, 591:19, 595:61, 601:48, 524:607, 611:247, 616:18, 618:70, 619:237, 620:204, 626:178, 629:73, 630:114, 634:286, 640:451, 642:396, 396:642, 645:602, 602:645, 650:251, 652:14, 656:102, 664:472, 666:114, 674:14, 678:137, 682:199, 683:[375,152], 686:[82,206], 688:[185,696], 185:688, 689:[204,620], 185:696, 555:696, 697:62, 698:423, 423:698, 699:285, 628:285, 627:285, 622:285, 561:285, 531:285, 520:285, 700:130, 703:503, 503:703, 705:[375,152], 712:61, 714:[316,751], 316:[714,751], 398:715, 571:715, 72:715, 717:114, 718:365, 365:718, 720:82, 726:173, 727:55, 55:727, 728:102, 729:237, 437:736, 290:[736,504], 504:736, 560:[736,504], 625:736, 738:82, 739:285, 574:746, 746:574, 747:178, 748:357, 749:185, 750:315, 751:[316,714], 752:114, 756:82, 757:285, 603:765, 768:[208,421], 773:736, 774:102, 775:736, 784:320, 788:285, 794:173, 798:736, 131:801, 803:607, 804:736, 806:[642,396], 642:396, 810:736, 811:70, 813:114, 822:413, 413:822, 824:[82, 32], 825:462, 462:825, 826:[17, 36], 830:102, 802:832, 832:802, 833:736, 842:736, 843:574,}

# pc-459
# father_class = {0: 261, 2: 135, 5: 87, 6: 43, 7: [60, 193], 14: 9, 15: 29, 23: 409, 25: 47, 28: 29, 33: 87, 34: 409, 35: [87, 397], 36: 87, 39: 43, 42: 87, 44: 58, 48: 161, 49: 409, 54: 87, 61: 87, 63: 238, 68: 135, 74: 397, 75: 409, 76: 409, 82: [135, 238], 84: 135, 89: 135, 91: 135, 95: 135, 99: 238, 100: 87, 110: [135, 238], 115: 135, 116: 135, 117: 135, 118: 238, 128: [135, 140], 129: [135, 409], 130: 135, 133: 238, 142: 135, 138: 145, 145: 138, 149: 87, 150: 87, 158: 292, 179: [36, 87], 186: 292, 190: 261, 199: 132, 201: 135, 228: 292, 274: [135, 238], 307: 292, 309: 135, 310: 292, 313: 135, 318: 135, 319: 135, 344: [135, 238], 384: 292, 389: 52, 406: 135, 419: 292, 428: [135, 238], 431: [135, 238], 432: [135, 238], 433: [135, 238], 434: 321, 435: 321, 442: [135, 238], 445: [135, 238], 60: 193, 193: 60, 423: 58, 162: 39, 216: 58, 132: 199, 276: 429, 303: 396, 366: 257, 361: 349, 352: 207, 372: 87, 387: 139, 350: [60, 193], 455: 397, 408: 397, 373: 397, 338: 397, 294: 397, 222: 397, 262: 397, 181: 180, 164: 397, 452: 71, 448: 171, 447: [438, 29], 438: [447, 29], 437: 261, 413: [412, 58], 71: 382, 382: 71, 427: 409, 426: 409, 421: 409, 369: 409, 368: 409, 340: 409, 322: 409, 298: 409, 192: 409, 94: 409, 69: 261, 441: 78, 78: [441, 45], 45: 78, 285: 261, 401: 79, 232: 231}


class GeneralizedSemSegEvaluator(SemSegEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        num_classes=None,
        ignore_label=None,
        post_process_func=None,
    ):
        super().__init__(
            dataset_name,
            distributed=distributed,
            output_dir=output_dir,
            num_classes=num_classes,
            ignore_label=ignore_label,
        )
        meta = MetadataCatalog.get(dataset_name)
        try:
            self._evaluation_set = meta.evaluation_set
        except AttributeError:
            self._evaluation_set = None
        self.post_process_func = (
            post_process_func
            if post_process_func is not None
            else lambda x, **kwargs: x
        )

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output = self.post_process_func(
                output["sem_seg"], image=np.array(Image.open(input["file_name"]))
            )
            output = output.argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=np.int)
            with PathManager.open(
                self.input_file_to_gt_file[input["file_name"]], "rb"
            ) as f:
                gt = np.array(Image.open(f), dtype=np.int)

            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=np.float)
        iou = np.full(self._num_classes, np.nan, dtype=np.float)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res["ACC-{}".format(name)] = 100 * acc[i]
        if self._evaluation_set is not None:
            for set_name, set_inds in self._evaluation_set.items():
                iou_list = []
                set_inds = np.array(set_inds, np.int)
                mask = np.zeros((len(iou),)).astype(np.bool)
                mask[set_inds] = 1
                miou = np.sum(iou[mask][acc_valid[mask]]) / np.sum(iou_valid[mask])
                pacc = np.sum(tp[mask]) / np.sum(pos_gt[mask])
                res["mIoU-{}".format(set_name)] = 100 * miou
                res["pAcc-{}".format(set_name)] = 100 * pacc
                iou_list.append(miou)
                miou = np.sum(iou[~mask][acc_valid[~mask]]) / np.sum(iou_valid[~mask])
                pacc = np.sum(tp[~mask]) / np.sum(pos_gt[~mask])
                res["mIoU-un{}".format(set_name)] = 100 * miou
                res["pAcc-un{}".format(set_name)] = 100 * pacc
                iou_list.append(miou)
                res["hIoU-{}".format(set_name)] = (
                    100 * len(iou_list) / sum([1 / iou for iou in iou_list])
                )
        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results
    

class SGIoU_SemSegEvaluator(SemSegEvaluator):
    """
    Evaluate semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        num_classes=None,
        ignore_label=None,
        post_process_func=None,
    ):
        super().__init__(
            dataset_name,
            distributed=distributed,
            output_dir=output_dir,
            num_classes=num_classes,
            ignore_label=ignore_label,
        )
        meta = MetadataCatalog.get(dataset_name)
        try:
            self._evaluation_set = meta.evaluation_set
        except AttributeError:
            self._evaluation_set = None
        self.post_process_func = (
            post_process_func
            if post_process_func is not None
            else lambda x, **kwargs: x
        )

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output = self.post_process_func(
                output["sem_seg"], image=np.array(Image.open(input["file_name"]))
            )
            output = output.argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=np.int)
            with PathManager.open(
                self.input_file_to_gt_file[input["file_name"]], "rb"
            ) as f:
                gt = np.array(Image.open(f), dtype=np.int)

            gt[gt == self._ignore_label] = self._num_classes    #(H, W)

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)  #(k+1, k+1)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))

    def evaluate(self):
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=np.float)
        iou = np.full(self._num_classes, np.nan, dtype=np.float)


        old_tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        tp = copy.deepcopy(old_tp)

        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)


        for cls_id in range(old_tp.shape[0]):
            if cls_id not in father_class.keys():
                continue
            else:
                # print(tp[cls_id], self._conf_matrix[father_class[cls_id]][cls_id])
                if isinstance(father_class[cls_id], list):
                    for father_pixel in father_class[cls_id]:
                        if pos_pred[father_pixel] > 0:
                            beta = (self._conf_matrix[father_pixel][cls_id] + self._conf_matrix[father_pixel][father_pixel]) / pos_pred[father_pixel]
                        else:
                            beta = 1
                        tp[cls_id] += self._conf_matrix[father_pixel][cls_id] * beta
                else:
                    if pos_pred[father_class[cls_id]] > 0 :
                        beta = (self._conf_matrix[father_class[cls_id]][cls_id] + self._conf_matrix[father_class[cls_id]][father_class[cls_id]]) / pos_pred[father_class[cls_id]]
                    else:
                        beta = 1
                    tp[cls_id] += self._conf_matrix[father_class[cls_id]][cls_id] * beta
        
        
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - old_tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res["ACC-{}".format(name)] = 100 * acc[i]
        if self._evaluation_set is not None:
            for set_name, set_inds in self._evaluation_set.items():
                iou_list = []
                set_inds = np.array(set_inds, np.int)
                mask = np.zeros((len(iou),)).astype(np.bool)
                mask[set_inds] = 1
                miou = np.sum(iou[mask][acc_valid[mask]]) / np.sum(iou_valid[mask])
                pacc = np.sum(tp[mask]) / np.sum(pos_gt[mask])
                res["mIoU-{}".format(set_name)] = 100 * miou
                res["pAcc-{}".format(set_name)] = 100 * pacc
                iou_list.append(miou)
                miou = np.sum(iou[~mask][acc_valid[~mask]]) / np.sum(iou_valid[~mask])
                pacc = np.sum(tp[~mask]) / np.sum(pos_gt[~mask])
                res["mIoU-un{}".format(set_name)] = 100 * miou
                res["pAcc-un{}".format(set_name)] = 100 * pacc
                iou_list.append(miou)
                res["hIoU-{}".format(set_name)] = (
                    100 * len(iou_list) / sum([1 / iou for iou in iou_list])
                )
        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results