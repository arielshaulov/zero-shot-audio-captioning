import argparse
import torch
from model.ZeroCLIP import CLIPTextGenerator
from model.ZeroCLIP_batched import CLIPTextGenerator as CLIPTextGenerator_multigpu
from models.imagebind_model import ModalityType
from pytube import YouTube
from datetime import datetime


def save_args_to_csv(f, args):
    for item in args.keys():
        value = args[item]
        f.write(str(item) + ',' + str(value) + '\n')
        f.flush()
    return f


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lm_model", type=str, default="gpt-2", help="gpt-2 or gpt-neo")
    parser.add_argument("--clip_checkpoints", type=str, default="./clip_checkpoints", help="path to CLIP")
    parser.add_argument("--target_seq_length", type=int, default=7)
    parser.add_argument("--cond_text", type=str, default="Audio of a")
    parser.add_argument("--reset_context_delta", action="store_true", default=True,
                        help="Should we reset the context at each token gen")
    parser.add_argument("--num_iterations", type=int, default=7)
    parser.add_argument("--clip_loss_temperature", type=float, default=0.01)
    parser.add_argument("--clip_scale", type=float, default=1)
    parser.add_argument("--ce_scale", type=float, default=0.2)
    parser.add_argument("--stepsize", type=float, default=0.3)
    parser.add_argument("--grad_norm_factor", type=float, default=0.9)
    parser.add_argument("--fusion_factor", type=float, default=0.99)
    parser.add_argument("--repetition_penalty", type=float, default=1)
    parser.add_argument("--end_token", type=str, default=".", help="Token to end text")
    parser.add_argument("--end_factor", type=float, default=1.01, help="Factor to increase end_token")
    parser.add_argument("--forbidden_factor", type=float, default=20, help="Factor to decrease forbidden tokens")
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--task", type=str, default='audio')
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument('--run_type',
                        default='caption',
                        nargs='?',
                        choices=['caption', 'arithmetics'])
    parser.add_argument("--caption_audio_path", type=str, default='audio/dog_audio.wav',
                        help="Path to audio for captioning")
    parser.add_argument("--arithmetics_weights", nargs="+", default=[1, 1, -1])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=3)
    parser.add_argument("--top_size", type=int, default=256)
    parser.add_argument("--w_guidence", type=float, default=1)
    args = parser.parse_args()
    return args

def run(args, audio_path):
    if args.multi_gpu:
        text_generator = CLIPTextGenerator_multigpu(**vars(args))
    else:
        text_generator = CLIPTextGenerator(**vars(args))
    audio_features = text_generator.get_audio_feature([audio_path], None)
    captions = text_generator.run(audio_features, args.cond_text, beam_size=args.beam_size)
    encoded_captions = [text_generator.get_txt_features([c]) for c in captions]
    encoded_captions = [d[ModalityType.TEXT] for d in encoded_captions]
    encoded_captions = [x / x.norm(dim=-1, keepdim=True) for x in encoded_captions]
    best_clip_idx = (audio_features[ModalityType.AUDIO] @ torch.cat(encoded_captions).t()).squeeze().argmax().item()
    return args.cond_text + captions[best_clip_idx]

if __name__ == "__main__":
    import glob
    from tqdm import tqdm
    import os
    import moviepy.editor as mp

    # Function to convert MP4 to WAV
    def convert_mp4_to_wav(mp4_file, wav_file):
        audio = mp.AudioFileClip(mp4_file)
        audio.write_audiofile(wav_file)
    
    # Function to cut audio from a start time (in seconds) to the end
    def cut_audio(wav_file, start_time, end_time, cut_output_file):
        audio = mp.AudioFileClip(wav_file)
        if audio.duration < start_time:
            return False
        if audio.duration > end_time:
            cut_audio = audio.subclip(start_time, end_time)
        else:
            cut_audio = audio.subclip(start_time)
        cut_audio.write_audiofile(cut_output_file)
        cut_audio.close()
        return True
    

    def get_ann(ann_file):
        with open(ann_file, 'r') as f:
            lines = f.readlines()
        d = {}
        for item in lines[1:]:
            item = item.strip().split(',')
            d[item[1]] = (item[2], item[3])
        return d
    
    def get_youtubeid():
        f1 = open('AudioCaps/id.csv', 'r')
        f2 = open('AudioCaps/name.csv', 'r')
        mp4_files = f1.readlines()
        id_files = f2.readlines()
        return id_files[:100], mp4_files[:100]

    def run_youtube(args):
        ann = get_ann('test.csv')
        now = datetime.now()
        name = now.strftime("%d_%m_%Y_%H_%M_%S")
        f = open('results/' + name + '.csv', 'w')
        save_args_to_csv(f, vars(args))
        f.write('file, caption, gt\n')
        f.flush()
        files = os.listdir('AudioCaps/')
        for _, file in enumerate(tqdm(files)):
            try:
                youtube_id = file.split('.wav')[0]
                wav_file = 'AudioCaps/' + file
                start_time = float(ann[youtube_id][0])
                gt = ann[youtube_id][1]
                flag = cut_audio(wav_file, start_time, start_time + 10, name + '.wav')
                if not flag:
                    continue
                caption = run(args, audio_path=name + '.wav')
                f.write(youtube_id + ',' + caption + ',' + gt + '\n')
                f.flush()
            except Exception as e:
                continue
            # os.remove(wav_file)
        f.close()

    def run_custom(args):
        files = glob.glob(args.task + '/*')[args.start:args.stop]
        f = open(args.task + '_' + str(args.start) + '_' + str(args.stop) + '.txt', 'w')
        f.write('file, caption\n')
        f.flush()
        for ix, wav_file in enumerate(tqdm(files)):
            caption = run(args, audio_path=wav_file)
            f.write(wav_file + ',' + caption + '\n')
            f.flush()
    args = get_args()
    # run_custom(args)
    run_youtube(args)

