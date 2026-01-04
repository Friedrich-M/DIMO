export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

input_folder=data/trump_n51_step20 # data folder
scene_name=$(basename $input_folder)
save_path=outputs/${scene_name} # checkpoint folder

if [ -d $save_path ]; then
    rm -r $save_path
fi

iters_s1=2800
iters_s2=10000

density_start_iter=200
density_end_iter=2000

densification_interval=100
densify_opacity_threshold_s1=0.02
densify_grad_threshold=0.02

arap_start_iter_s1=2000
arap_end_iter_s2=5000

num_cpts=512

num_frames=21
ref_size=512
batch_size=2
latent_code_dim=32

python main_train_dimo.py train_dynamic=True \
    input_folder=$input_folder save_path=$save_path \
    num_frames=$num_frames ref_size=$ref_size batch_size=$batch_size \
    num_cpts=$num_cpts latent_code_dim=$latent_code_dim \
    iters_s1=$iters_s1 iters_s2=$iters_s2 \
    densification_interval=$densification_interval \
    densify_opacity_threshold_s1=$densify_opacity_threshold_s1 \
    densify_grad_threshold=$densify_grad_threshold \
    density_start_iter=$density_start_iter density_end_iter=$density_end_iter \
    arap_start_iter_s1=$arap_start_iter_s1 arap_end_iter_s2=$arap_end_iter_s2



