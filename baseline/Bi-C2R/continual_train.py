from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys
import os
import copy
import torch

from torch.backends import cudnn
import torch.nn as nn
import random
from config import cfg
from reid.evaluators import Evaluator
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.lr_scheduler import WarmupMultiStepLR
from reid.utils.feature_tools import *
from reid.models.layers import DataParallel
from reid.models.resnet import make_model, TransNet_adaptive
from reid.trainer import Trainer
from reid.utils.fisher import estimate_fisher, normalize_fisher, merge_fisher, save_fisher_checkpoint, snapshot_parameters, fisher_aware_state_dict_fusion
from torch.utils.tensorboard import SummaryWriter

from lreid_dataset.datasets.get_data_loaders import build_data_loaders
from tools.Logger_results import Logger_res

import warnings
warnings.filterwarnings("ignore")

def worker_init_fn(worked_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    args = parser.parse_args()

    # Fisher-aware DFF 必须依赖上一阶段 Fisher；显式校验可避免静默回退。
    if args.use_fisher_dff and not args.use_fisher:
        parser.error("Fisher-aware DFF 依赖历史 Fisher，请同时指定 --use-fisher")
    if args.fisher_dff_beta < 0:
        parser.error("--fisher-dff-beta 不能小于 0")

    if args.seed is not None:
        np.random.seed(args.seed) 
        random.seed(args.seed)
        
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.manual_seed(args.seed)
        
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False


    cfg.merge_from_file(args.config_file)
    main_worker(args, cfg)


def main_worker(args, cfg):
    log_name = 'log.txt'
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    else:
        log_dir = osp.dirname(args.test_folder)
        sys.stdout = Logger(osp.join(log_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))
    log_res_name='log_res.txt'
    logger_res=Logger_res(osp.join(args.logs_dir, log_res_name))
    

    """
    loading the datasets:
    setting閿?1 or 2 
    """
    if 1 == args.setting:
        training_set = ['market1501', 'cuhk_sysu', 'dukemtmc', 'msmt17', 'cuhk03']
    elif 2 == args.setting:
        training_set = ['dukemtmc', 'msmt17', 'market1501', 'cuhk_sysu', 'cuhk03']
    
    all_set = ['market1501', 'dukemtmc', 'msmt17', 'cuhk_sysu', 'cuhk03',
               'cuhk01', 'cuhk02', 'grid', 'sense', 'viper', 'ilids', 'prid']

    testing_only_set = [x for x in all_set if x not in training_set]
    all_train_sets, all_test_only_sets = build_data_loaders(args, training_set, testing_only_set)    
    
    first_train_set = all_train_sets[0]
    model = make_model(args, num_class=first_train_set[1], camera_num=0, view_num=0)
    model_trans=TransNet_adaptive()
    model_trans2=TransNet_adaptive()

    model.cuda()
    model_trans.cuda()
    model_trans2.cuda()
    
    model = DataParallel(model)    
    model_trans = DataParallel(model_trans)
    model_trans2 = DataParallel(model_trans2)

    writer = SummaryWriter(log_dir=args.logs_dir)
    '''test the models under a folder'''
    if args.test_folder:
        ckpt_name = [x + '_checkpoint.pth.tar' for x in training_set]
        checkpoint = load_checkpoint(osp.join(args.test_folder, ckpt_name[0]))
        copy_state_dict(checkpoint['state_dict'], model)
        for step in range(len(ckpt_name) - 1):
            model_old = copy.deepcopy(model)        
            checkpoint = load_checkpoint(osp.join(args.test_folder, ckpt_name[step + 1]))
            copy_state_dict(checkpoint['state_dict'], model)

                         
            best_alpha = get_adaptive_alpha(args, model, model_old, all_train_sets, step + 1)
            model = linear_combination(args, model, model_old, best_alpha)
            save_name = '{}_checkpoint_adaptive_ema_{:.4f}.pth.tar'.format(training_set[step+1], best_alpha)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': 0,
                'mAP': 0,
            }, True, fpath=osp.join(args.logs_dir, save_name))
        test_model(model, all_train_sets, all_test_only_sets, len(all_train_sets)-1, logger_res=logger_res, feats_dir=args.logs_dir)

        exit(0)
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['mAP']
        print("=> Start epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))

    if args.MODEL in ['50x']:
        out_channel = 2048
    else:
        raise AssertionError(f"the model {args.MODEL} is not supported!")

    previous_fisher = None
    for set_index in range(0, len(training_set)):
        # classifier 扩展前保存上一阶段部署参数，作为本阶段 Fisher 固化锚点。
        # old_model 继续用于 Bi-C2R 蒸馏，两者职责相互独立。
        old_params = snapshot_parameters(model) if set_index > 0 else None
        old_fisher = previous_fisher if set_index > 0 else None
        model_old = copy.deepcopy(model)
        model, model_trans, model_trans2 = train_dataset(
            cfg, args, all_train_sets, all_test_only_sets, set_index,
            model, model_trans, model_trans2, out_channel, writer,
            logger_res=logger_res, old_params=old_params, old_fisher=old_fisher,
        )

        if set_index > 0:
            best_alpha = get_adaptive_alpha(args, model, model_old, all_train_sets, set_index)
            if args.use_fisher and args.use_fisher_dff and old_fisher is not None:
                # 原始 DFF 的全局 alpha 表示阶段级兼容性；Historical Fisher
                # 将其细化为参数级融合系数，使重要历史参数更接近旧模型。
                fused_state = fisher_aware_state_dict_fusion(
                    model.state_dict(),
                    model_old.state_dict(),
                    old_fisher,
                    alpha=best_alpha,
                    beta=args.fisher_dff_beta,
                )
                model.load_state_dict(fused_state)
                print("Applied Fisher-aware DFF: alpha={:.6f}, beta={:.6f}".format(
                    best_alpha, args.fisher_dff_beta
                ))
            else:
                model = linear_combination(args, model, model_old, best_alpha)

        dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[set_index]

        # 必须在 DFF 完成后统计 Fisher，确保重要性矩阵与最终部署模型一致。
        if args.use_fisher:
            fisher_new = estimate_fisher(
                model=model,
                dataloader=init_loader,
                device=next(model.parameters()).device,
                max_batches=args.fisher_max_batches,
                label_offset=sum(all_train_sets[i][1] for i in range(set_index)),
                estimator=args.fisher_estimator,
                protect_classifier=True,
            )
            fisher_new = normalize_fisher(
                fisher_new,
                mode=args.fisher_norm,
                clip_value=args.fisher_clip_value,
            )
            previous_fisher = merge_fisher(
                previous_fisher or {}, fisher_new, gamma=args.fisher_gamma
            )
            fisher_path = osp.join(args.logs_dir, '{}_fisher.pth'.format(name))
            save_fisher_checkpoint(
                fisher_path,
                previous_fisher,
                stage=set_index + 1,
                class_mapping={
                    'old_class_count': int(sum(all_train_sets[i][1] for i in range(set_index)))
                },
                metadata={
                    'estimator': args.fisher_estimator,
                    'normalization': args.fisher_norm,
                    'gamma': args.fisher_gamma,
                    'source': 'stage_data_and_post_dff_model',
                },
            )
            # 模型 checkpoint 与 Fisher 使用同一阶段编号，保证可追溯性。
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': 0,
                'mAP': 0.,
                'fisher_path': fisher_path,
            }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))
            print('Saved Historical Fisher: {}'.format(fisher_path))
        from reid.evaluators import extract_features
        features, _ = extract_features(model, test_loader, training_phase=set_index+1)
        print ("Save Features: ", len(features))
        torch.save({'features':features}, args.logs_dir+name+'_features.pth.tar')
        
        if args.trans_feat and set_index > 0:
            for each_data in training_set[0:set_index]:
                each_old_gallery = torch.load(args.logs_dir + str(each_data) + '_features.pth.tar')
                features_dict = each_old_gallery['features']
                feature_keys = list(features_dict.keys())
                feature_values = list(features_dict.values())
                feature_tensor = torch.stack([torch.tensor(val) for val in feature_values])
                feature_tensor = F.normalize(feature_tensor, p=2, dim=1)
                from collections import OrderedDict
                updated_features_dict = OrderedDict()
                model_trans.eval()                
                with torch.no_grad():
                    
                    for start_pos in range(0, feature_tensor.shape[0], args.batch_size):
                        end_pos = min(start_pos + args.batch_size, feature_tensor.shape[0])
                        batch_features = feature_tensor[start_pos:end_pos]
                        batch_features_trans = model_trans(batch_features)
                        batch_features_trans = F.normalize(batch_features_trans, p=2, dim=1)
                        batch_features_trans = best_alpha * batch_features_trans + (1 - best_alpha) * batch_features.cuda()
                        for i in range(start_pos, end_pos):
                            updated_features_dict[feature_keys[i]] = batch_features_trans[i - start_pos]
                    each_old_gallery['features'] = updated_features_dict
                    torch.save(each_old_gallery, args.logs_dir + str(each_data) + '_features.pth.tar')
                    print("Old Gallery Feature Updated to: ", args.logs_dir + str(each_data) + '_features.pth.tar')
            print('=================================================================================')
        if args.set_zero:
            model_trans = set_zero(model_trans)
            model_trans2 = set_zero(model_trans2)
        if set_index>0:
            test_model(model, all_train_sets, all_test_only_sets, set_index, logger_res=logger_res, feats_dir=args.logs_dir)    
    print('finished')
def get_normal_affinity(x,Norm=100):
    from reid.metric_learning.distance import cosine_similarity
    pre_matrix_origin=cosine_similarity(x,x)
    pre_affinity_matrix=F.softmax(pre_matrix_origin*Norm, dim=1)
    return pre_affinity_matrix
def get_adaptive_alpha(args, model, model_old, all_train_sets, set_index):
    dataset_new, num_classes_new, train_loader_new, _, init_loader_new, name_new = all_train_sets[
        set_index]
    features_all_new, labels_all, fnames_all, camids_all, features_mean_new, labels_named = extract_features_voro(model,
                                                                                                          init_loader_new,
                                                                                                          get_mean_feature=True)
    features_all_old, _, _, _, features_mean_old, _ = extract_features_voro(model_old,init_loader_new,get_mean_feature=True)

    features_all_new=torch.stack(features_all_new, dim=0)
    features_all_old=torch.stack(features_all_old,dim=0)
    Affin_new = get_normal_affinity(features_all_new)
    Affin_old = get_normal_affinity(features_all_old)

    Difference= torch.abs(Affin_new-Affin_old).sum(-1).mean()

    alpha=float(1-Difference)
    return alpha



def train_dataset(cfg, args, all_train_sets, all_test_only_sets, set_index, model, model_trans, model_trans2, out_channel, writer,logger_res=None, old_params=None, old_fisher=None):
    dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[
        set_index]

    Epochs= args.epochs0 if 0==set_index else args.epochs          

    if set_index<=1:
        add_num = 0
        old_model=None
    else:
        add_num = sum(
            [all_train_sets[i][1] for i in range(set_index - 1)])
    
    
    if set_index>0:
        '''store the old model'''
        old_model = copy.deepcopy(model)
        old_model = old_model.cuda()
        old_model.eval()

        add_num = sum([all_train_sets[i][1] for i in range(set_index)])

        org_classifier_params = model.module.classifier.weight.data
        model.module.classifier = nn.Linear(out_channel, add_num + num_classes, bias=False)
        model.module.classifier.weight.data[:add_num].copy_(org_classifier_params)
        model.cuda()    

        class_centers = initial_classifier(model, init_loader)
        model.module.classifier.weight.data[add_num:].copy_(class_centers)
        model.cuda()

    params = []
    for key, value in model.named_params(model):
        if not value.requires_grad:
            print('not requires_grad:', key)
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    if set_index>0:
        for key, value in model_trans.named_params(model_trans):
            if not value.requires_grad:
                print('not requires_grad:', key)
                continue
            params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
        for key, value in model_trans2.named_params(model_trans2):
            if not value.requires_grad:
                print('not requires_grad:', key)
                continue
            params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=args.momentum)    
    Stones=args.milestones
    lr_scheduler = WarmupMultiStepLR(optimizer, Stones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)
    
  
    trainer = Trainer(cfg, args, model, model_trans, model_trans2, add_num + num_classes,  writer=writer)

    print('####### starting training on {} #######'.format(name))
    for epoch in range(0, Epochs):

        train_loader.new_epoch()
        trainer.train(epoch, train_loader, optimizer, training_phase=set_index + 1,
                      train_iters=len(train_loader), add_num=add_num, old_model=old_model,
                      old_params=old_params, old_fisher=old_fisher,
                      )
        lr_scheduler.step()       
       

        if ((epoch + 1) % args.eval_epoch == 0 or epoch+1==Epochs):
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': 0.,
            }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))

            logger_res.append('epoch: {}'.format(epoch + 1))
            
            mAP=0.
            if args.middle_test:
                mAP = test_model(model, all_train_sets, all_test_only_sets, set_index, logger_res=logger_res, feats_dir=args.logs_dir)                    
          
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': mAP,
            }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))    

    return model, model_trans, model_trans2

def test_model(model, all_train_sets, all_test_sets, set_index, logger_res=None, feats_dir=None):
    begin = 0
    evaluator = Evaluator(model)
        
    R1_all = []
    mAP_all = []
    names=''
    Results=''
    for i in range(begin, set_index + 1):
        dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[i]
        print('Results on {}'.format(name))

        train_R1, train_mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery,
                                                 cmc_flag=True)
        R1_all.append(train_R1)
        mAP_all.append(train_mAP)
        names = names + name + '\t\t'
        Results=Results+'|{:.1f}/{:.1f}\t'.format(train_mAP* 100, train_R1* 100)

    aver_mAP = torch.tensor(mAP_all).mean()
    aver_R1 = torch.tensor(R1_all).mean()


    R1_all = []
    mAP_all = []
    names_unseen = ''
    Results_unseen = ''
    for i in range(begin, set_index + 1):
        dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[i]
        print('Results on {}'.format(name))

        R1, mAP = evaluator.evaluate_rfl(test_loader, dataset.query, dataset.gallery,
                                     cmc_flag=True, old_feat=feats_dir+name+'_features.pth.tar')
        R1_all.append(R1)
        mAP_all.append(mAP)
        names_unseen = names_unseen + name + '\t'
        Results_unseen = Results_unseen + '|{:.1f}/{:.1f}\t'.format(mAP* 100, R1* 100)

    aver_mAP_unseen = torch.tensor(mAP_all).mean()
    aver_R1_unseen = torch.tensor(R1_all).mean()

    print("Average mAP on Seen dataset: {:.1f}%".format(aver_mAP * 100))
    print("Average R1 on Seen dataset: {:.1f}%".format(aver_R1 * 100))
    names = names + '|Average\t|'
    Results = Results + '|{:.1f}/{:.1f}\t|'.format(aver_mAP * 100, aver_R1 * 100)
    print(names)
    print(Results)
    '''_________________________'''
    print("Average mAP on unSeen dataset: {:.1f}%".format(aver_mAP_unseen * 100))
    print("Average R1 on unSeen dataset: {:.1f}%".format(aver_R1_unseen * 100))
    names_unseen = names_unseen + '|Average\t|'
    Results_unseen = Results_unseen + '|{:.1f}/{:.1f}\t|'.format(aver_mAP_unseen* 100, aver_R1_unseen* 100)
    print(names_unseen)
    print(Results_unseen)
    if logger_res:
        logger_res.append(names)
        logger_res.append(Results)
        logger_res.append(Results.replace('|','').replace('/','\t'))
        logger_res.append(names_unseen)
        logger_res.append(Results_unseen)
        logger_res.append(Results_unseen.replace('|', '').replace('/', '\t'))
    return train_mAP



def linear_combination(args, model, model_old, alpha, model_old_id=-1):
    '''old model '''
    model_old_state_dict = model_old.state_dict()
    '''latest trained model'''
    model_state_dict = model.state_dict()

    ''''create new model'''
    model_new = copy.deepcopy(model)
    model_new_state_dict = model_new.state_dict()
    '''fuse the parameters'''
    for k, v in model_state_dict.items():
        if model_old_state_dict[k].shape == v.shape:
                model_new_state_dict[k] = alpha * v + (1 - alpha) * model_old_state_dict[k]
        else:
            print(k, '...')
            num_class_old = model_old_state_dict[k].shape[0]
            model_new_state_dict[k][:num_class_old] = alpha * v[:num_class_old] + (1 - alpha) * model_old_state_dict[k]
    model_new.load_state_dict(model_new_state_dict)
    return model_new

def set_zero(model):
    '''old model '''
    '''latest trained model'''
    model_state_dict = model.state_dict()

    ''''create new model'''
    model_new = copy.deepcopy(model)
    model_new_state_dict = model_new.state_dict()
    '''fuse the parameters'''
    for k, v in model_state_dict.items():
        model_new_state_dict[k] = 0. * v
    
    model_new.load_state_dict(model_new_state_dict)
    return model_new

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")

    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
 
    parser.add_argument('--MODEL', type=str, default='50x',
                        choices=['50x'])

    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'],
                        help="optimizer ")
    parser.add_argument('--lr', type=float, default=0.008,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[30],
                        help='milestones for the learning rate decay')

    parser.add_argument('--resume', type=str, default=None, metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs0', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--eval_epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=24)
    parser.add_argument('--print-freq', type=int, default=200)
    
 
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/home/xiong_project/data/PRID/')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join('../logs/try'))

    parser.add_argument('--config_file', type=str, default='config/base.yml',
                        help="config_file")
  
    parser.add_argument('--test_folder', type=str, default=None, help="test the models in a file")
   
    parser.add_argument('--setting', type=int, default=1, choices=[1, 2], help="training order setting")
    parser.add_argument('--middle_test', action='store_true', help="test during middle step")
    parser.add_argument('--AF_weight', default=1.0, type=float, help="anti-forgetting weight")    
    
    parser.add_argument('--trans_feat', action="store_false", help='if or not transform old features')
    parser.add_argument('--set_zero', action="store_true", help='if or not set 0 to transnet')
    parser.add_argument('--weight_trans', type=float, default=100, help='weight for transformation loss')
    parser.add_argument('--weight_anti', type=float, default=1, help='weight for anti_forget loss')
    parser.add_argument('--weight_discri', type=float, default=0.007, help='weight for anti_discrimination loss')
    parser.add_argument('--weight_transx', type=float, default=0.0005, help='weight for transformation_x loss')
    # Historical Fisher 配置默认关闭，保证原始 Bi-C2R 实验路径不变。
    parser.add_argument("--use-fisher", action="store_true",
                        help="启用 Historical Fisher 参数固化")
    parser.add_argument("--lambda-fim", type=float, default=1e-5,
                        help="Fisher 固化损失权重")
    parser.add_argument("--fisher-gamma", type=float, default=1.0,
                        help="历史 Fisher 在线累计系数")
    parser.add_argument("--fisher-estimator", type=str, default="per_sample",
                        choices=["per_sample", "batch_mean"],
                        help="Fisher 梯度平方估计方式")
    parser.add_argument("--fisher-norm", type=str, default="per_parameter_mean",
                        choices=["none", "global_mean", "per_parameter_mean"],
                        help="Fisher 归一化方式")
    parser.add_argument("--fisher-clip-value", type=float, default=None,
                        help="Fisher 最大截断值；默认不截断")
    parser.add_argument("--fisher-max-batches", type=int, default=None,
                        help="Fisher 统计最多使用的 batch 数；默认使用全部 batch")
    parser.add_argument("--use-fisher-dff", action="store_true",
                        help="使用 Fisher-aware DFF 进行参数级模型融合")
    parser.add_argument("--fisher-dff-beta", type=float, default=1.0,
                        help="Fisher-aware DFF 的参数重要性调节系数")

    main()
