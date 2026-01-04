export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

input_folder=data/trump_n51_step20 # data folder
scene_name=$(basename $input_folder)
save_path=ckpts/${scene_name} # checkpoint folder
video_save_dir=vis/${scene_name} # save rendering results

if [ -d $video_save_dir ]; then
    rm -r $video_save_dir
fi

test_stage=s2
num_frames=21
ref_size=512

python main_test_dimo.py test_paper=True \
    input_folder=$input_folder save_path=$save_path video_save_dir=$video_save_dir \
    test_azi=0 num_frames=$num_frames ref_size=$ref_size test_stage=$test_stage \
    render_videos=11-walk # choose which motion to render; comment out to render all motions
