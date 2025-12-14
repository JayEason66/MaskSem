export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=4,5,6,7


# NTU60 xview
python -m torch.distributed.launch --nproc_per_node=4 --master_port 11234 main_pretrain.py \
--config ./config/ntu60_xview_joint/pretrain_act_t120_layer8+3_mask90.yaml \
--output_dir ./output_dir2/ntu60_xview_joint/pretrain_act_t120_layer8+3_binary_CAM_mask90_tau0.80_ep400_noamp \
--finetune /MAMP/output_dir/ntu60_xview_joint/pretrain_mamp_t120_layer8+3_mask90_tau0.80_ep400_noamp/checkpoint-399.pth \
--log_dir ./output_dir2/ntu60_xview_joint/pretrain_act_t120_layer8+3_binary_CAM_mask90_tau0.80_ep400_noamp