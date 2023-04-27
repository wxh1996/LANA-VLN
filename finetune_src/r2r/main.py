from datetime import datetime
import os
import json
import time
import numpy as np
from collections import defaultdict

import torch
from tensorboardX import SummaryWriter
import datetime
import sys

sys.path.append(os.getcwd())

from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results

from models.vlnbert_init_lana import get_tokenizer

from r2r.agent_cmt_lana import Seq2SeqCMTAgent

from r2r.data_utils import ImageFeaturesDB, construct_instrs
from r2r.env import R2RBatch, R2RBackBatch
from r2r.parser import parse_args
from r2r.cap_eval.COCOEvalCap import COCOEvalCap
from models.tokenization_clip import SimpleTokenizer
from tqdm import tqdm

def build_dataset(args, rank=0, is_test=False):
    tok = get_tokenizer(args)

    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size, use_clip16=args.use_clip16)

    if args.dataset == 'r2r_back':
        dataset_class = R2RBackBatch
    else:
        dataset_class = R2RBatch

    # because we don't use distributed sampler here
    # in order to make different processes deal with different training examples
    # we need to shuffle the data with different seed in each processes
    train_env_name = 'train' if not args.debug else 'val_unseen'
    train_instr_data = construct_instrs(
        args.anno_dir, args.dataset, [train_env_name], tokenizer=tok, max_instr_len=args.max_instr_len, use_clip16=args.use_clip16
    )
    train_env = dataset_class(
        feat_db, train_instr_data, args.connectivity_dir, batch_size=args.batch_size,
        angle_feat_size=args.angle_feat_size, seed=args.seed + rank,
        sel_data_idxs=None, name=train_env_name, anno_dir=args.anno_dir, is_reverie=(args.dataset == 'reverie')
    )
    if args.aug is not None and not args.debug:
        aug_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [args.aug], tokenizer=tok, max_instr_len=args.max_instr_len, use_clip16=args.use_clip16
        )
        aug_env = dataset_class(
            feat_db, aug_instr_data, args.connectivity_dir, batch_size=args.batch_size,
            angle_feat_size=args.angle_feat_size, seed=args.seed + rank,
            sel_data_idxs=None, name='aug', anno_dir=args.anno_dir, is_reverie=(args.dataset == 'reverie')
        )
    else:
        aug_env = None

    val_env_names = ['val_train_seen', 'val_seen']
    if args.test or args.dataset != 'r4r' or (args.dataset == 'r4r' and args.r4r_testall):
        val_env_names.append('val_unseen')
        # val_env_names.append('val_unseen_sampled')
    else:  # val_unseen of r4r is too large to evaluate in training
        val_env_names.append('val_unseen_sampled')

    if args.submit:
        if args.dataset == 'r2r':
            val_env_names.append('test')
        elif args.dataset == 'rxr':
            val_env_names.extend(['test_challenge_public', 'test_standard_public'])

    if args.debug:
        if args.dataset == 'r2r':
            val_env_names = ['val_seen', 'val_unseen']
        if args.dataset == 'r4r':
            val_env_names = ['val_unseen_sampled']

    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split], tokenizer=tok, max_instr_len=args.max_instr_len, use_clip16=args.use_clip16
        )
        val_env = dataset_class(
            feat_db, val_instr_data, args.connectivity_dir, batch_size=args.batch_size,
            angle_feat_size=args.angle_feat_size, seed=args.seed + rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split,
            anno_dir=args.anno_dir, is_reverie=(args.dataset == 'reverie')
        )
        evaluator = COCOEvalCap([split], tok, val_instr_data, use_clip16=args.use_clip16)
        val_envs[split] = (val_env, evaluator, split)

    return train_env, val_envs, aug_env


def train(args, train_env, val_envs, aug_env=None, rank=-1):
    default_gpu = is_default_gpu(args)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    agent_class = Seq2SeqCMTAgent
    listener = agent_class(args, train_env, rank=rank)

    # resume file
    start_iter = 0
    if args.resume_file is not None:
        start_iter = listener.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, iteration {}".format(args.resume_file, start_iter),
                record_file
            )

    # first evaluation
    print(datetime.datetime.now())

    if args.eval_first:
        loss_str = "validation before training"
        for env_name, (env, evaluator, name) in val_envs.items():
            if name == "val_train_seen":
                continue
            listener.env = env
            # Get validation distance from goal under test evaluation conditions
            listener.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listener.get_results()
            # gather distributed results
            preds = merge_dist_results(all_gather(preds))
            if default_gpu:
                score_summary, _, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
        if default_gpu:
            write_to_record_file(loss_str, record_file)
            
        print('test language metric')
        for env_name, (env, evaluator, name) in val_envs.items():
            if name == "val_train_seen":
                continue
            listener.env = env
            path2inst = listener.valid_speaker()
            evaluator.evaluate(path2inst)
            eval_str = evaluator.eval.items()
            write_to_record_file(str(eval_str), record_file)
    print(datetime.datetime.now())

    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start iteration: %s' % str(start_iter), record_file
        )

    if args.dataset == 'r4r' and not args.r4r_testall:
        best_val = {'val_unseen_sampled': {"spl": 0., "sr": 0., "state": ""}}
    else:
        best_val = {'val_unseen': {"spl": 0., "sr": 0., "state": ""}}

    for idx in range(start_iter, start_iter + args.iters, args.log_every):
        listener.logs = defaultdict(list)
        interval = min(args.log_every, args.iters - idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:
            task = args.task_name
            contrastive_loss = 0.0
            caption_loss = 0.0
            if task == "caption":
                jdx_length = len(range(interval // 2))
                for jdx in range(interval // 2):
                    listener.env = train_env
                    caption_loss = listener.train_speaker(1)  
                    if default_gpu:
                        print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)
            elif task == "vln":
                jdx_length = len(range(interval // 2))
                for jdx in range(interval // 2):
                    tasks = ['caption', 'regular']
                    weights = torch.Tensor([args.caption_task_weight, args.vln_task_weight])
                    task_id = torch.multinomial(weights, 1, replacement=True)
                    if task_id == 0:
                        _task = 'caption'
                    elif task_id == 1:
                        _task = 'regular'
                    if _task == "caption":
                        listener.env = train_env
                        caption_loss = listener.train_speaker(1)
                    elif _task == "regular":
                        listener.env = train_env
                        listener.train(1, feedback=args.feedback)  # Train interval iters
                    if default_gpu:
                        print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)
        else:
            contrastive_loss = 0.0
            caption_loss = 0.0
            jdx_length = len(range(interval // 2))
            for jdx in range(interval // 2):
                weights = torch.Tensor([args.caption_task_weight, args.vln_task_weight])
                task_id = torch.multinomial(weights, 1, replacement=True)
                if task_id == 0:
                    task = 'caption'
                elif task_id == 1:
                    task = 'regular'
                
                if task == "caption":
                    listener.env = train_env
                    caption_loss = listener.train_speaker(1)
                elif task == "contrastive":
                    listener.env = train_env
                    contrastive_loss1 = listener.train_cont(1)
                    listener.env = aug_env
                    contrastive_loss2 = listener.train_cont(1)
                    contrastive_loss = (contrastive_loss1 + contrastive_loss2) / 2.0
                else:
                    # Train with GT data
                    listener.env = train_env
                    listener.train(1, feedback=args.feedback)
                    # Train with Augmented data
                    listener.env = aug_env
                    listener.train(1, feedback=args.feedback)

                if default_gpu:
                    print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        if default_gpu:
            # Log the training stats to tensorboard
            total = max(sum(listener.logs['total']), 1)  # RL: total valid actions for all examples in the batch
            length = max(len(listener.logs['critic_loss']), 1)  # RL: total (max length) in the batch
            critic_loss = sum(listener.logs['critic_loss']) / total
            policy_loss = sum(listener.logs['policy_loss']) / total
            RL_loss = sum(listener.logs['RL_loss']) / max(len(listener.logs['RL_loss']), 1)
            IL_loss = sum(listener.logs['IL_loss']) / max(len(listener.logs['IL_loss']), 1)
            GP_loss = sum(listener.logs['GP_loss']) / max(len(listener.logs['GP_loss']), 1)
            entropy = sum(listener.logs['entropy']) / total
            writer.add_scalar("loss/critic", critic_loss, idx)
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/RL_loss", RL_loss, idx)
            writer.add_scalar("loss/IL_loss", IL_loss, idx)
            writer.add_scalar("loss/GP_loss", GP_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)
            write_to_record_file(
                "\ntotal_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, RL_loss %.4f, GP_loss %.4f,"
                " policy_loss %.4f, "
                "critic_loss %.4f, caption_loss %.4f, contrastive_loss %.4f" % (
                    total, length, entropy, IL_loss, RL_loss, GP_loss, policy_loss, critic_loss, caption_loss, contrastive_loss),
                record_file
            )

        # Run validation
        if args.task_name == "caption":
            print('test language metric')
            for env_name, (env, evaluator, name) in val_envs.items():
                if name == "val_train_seen":
                    continue
                listener.env = env
                path2inst = listener.valid_speaker()
                evaluator.evaluate(path2inst)
                eval_str = evaluator.eval.items()
                write_to_record_file(str(eval_str), record_file)
            continue
        
        loss_str = "iter {}".format(iter)
        for env_name, (env, evaluator, name) in val_envs.items():
            if name == "val_train_seen" or name == 'val_seen':
                continue
            listener.env = env

            # Get validation distance from goal under test evaluation conditions
            listener.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listener.get_results()
            preds = merge_dist_results(all_gather(preds))

            if default_gpu:
                score_summary, _, _ = env.eval_metrics(preds)
                loss_str += "\n %18s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], idx)

                # select model by only sr
                if env_name in best_val:
                    if score_summary['sr'] + score_summary['spl'] > best_val[env_name]['spl'] + best_val[env_name]['sr']:
                        best_val[env_name]['spl'] = score_summary['spl']
                        best_val[env_name]['sr'] = score_summary['sr']
                        best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                        listener.save(idx, os.path.join(args.ckpt_dir, "best_%s" % env_name))
                    if args.multiple_saving and score_summary['sr'] > args.baseline_sr:
                        listener.save(idx, os.path.join(args.ckpt_dir, "best_%s_%s" % (env_name, iter)))

        if default_gpu:
            listener.save(idx, os.path.join(args.ckpt_dir, "latest_dict"))

            write_to_record_file(
                ('%s (%d %d%%) %s' % (
                timeSince(start, float(iter) / args.iters), iter, float(iter) / args.iters * 100, loss_str)),
                record_file
            )
            write_to_record_file("BEST RESULT TILL NOW", record_file)
            for env_name in best_val:
                write_to_record_file(env_name + ' | ' + best_val[env_name]['state'], record_file)


def valid(args, train_env, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)

    agent_class = Seq2SeqCMTAgent
    agent = agent_class(args, train_env, rank=rank)


    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (agent.load(args.resume_file), args.resume_file))

    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    # if args.dataset == 'r4r':
    if args.dataset == 'r4r' or args.dataset == 'r2r':
        print('test language metric')
        for env_name, (env, evaluator, name) in val_envs.items():
            if name == "val_train_seen" or name == "test":
                continue
            agent.env = env
            path2inst = agent.valid_speaker()
            evaluator.evaluate(path2inst)
            eval_str = evaluator.eval.items()
            write_to_record_file(str(eval_str), record_file)
        # return
            
    for env_name, (env, evaluator, name) in val_envs.items():
        # if os.path.exists(os.path.join(args.pred_dir, "submit_%s.json" % env_name)):
        #     continue
        if name == "val_train_seen":
            continue
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        start_time = time.time()
        agent.test(use_dropout=False, feedback=args.test_feedback, iters=iters)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        preds = agent.get_results()
        preds = merge_dist_results(all_gather(preds))

        if default_gpu:
            if 'test' not in env_name:
                score_summary, _, sr_dicts_better = env.eval_metrics(preds)
                loss_str = "Env name: %s" % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                write_to_record_file(loss_str + '\n', record_file)
                
            if args.submit:
                json.dump(
                    preds,
                    open(os.path.join(args.pred_dir, "submit_%s.json" % env_name), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )
        
def main():
    args = parse_args()

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)
    train_env, val_envs, aug_env = build_dataset(args, rank=rank)

    if not args.test:
        train(args, train_env, val_envs, aug_env=aug_env, rank=rank)
    else:
        valid(args, train_env, val_envs, rank=rank)


if __name__ == '__main__':
    main()
