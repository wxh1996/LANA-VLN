ob_type=pano
feedback=sample

features=clip16  # or vitbase
ft_dim=512

ngpus=1
seed=1

outdir=path_to_your_output

flag="
      --rl_teacher_weight 0.4

      --root_dir ../datasets
      --output_dir ${outdir}

      --dataset r4r

      --vlnbert ${vlnbert}
      --ob_type ${ob_type}

      --world_size ${ngpus}
      --seed ${seed}
      
      --hist_enc_pano
      --hist_pano_num_layers 2

      --features ${features}
      --feedback ${feedback}

      --max_action_len 15
      --max_instr_len 100

      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 1e-5
      --iters 200000
      --log_every 1000
      --batch_size 8
      --optim adamW

      --ml_weight 0.2

      --feat_dropout 0.4
      --dropout 0.5
      --clip_lr 1e-5 
      --vln_task_weight 3 
      --caption_task_weight 1 "

CUDA_VISIBLE_DEVICES='0' python3 r2r/main.py $flag \
      --root_dir path_to_your_dataset \
      --use_clip16 \
      --act_pred_token ob \
      --bert_ckpt_file path_to_your_pretrained_checkpoint \
