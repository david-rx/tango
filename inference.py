
import scipy
import os
import copy
import json
import time
import torch
import argparse
import soundfile as sf
import wandb
from tqdm import tqdm
from diffusers import DDPMScheduler
from audioldm_eval import EvaluationHelper
from models import build_pretrained_models, AudioDiffusion
from transformers import AutoProcessor, ClapModel
import torchaudio
from tango import Tango
from diffusers import AudioLDMPipeline

USE_AUDIOLDM_VAE = False
MATCH_AUDIOLDM = True
USE_CUDA = False

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for text to audio generation task.")
    parser.add_argument(
        "--original_args", type=str, default=None,
        help="Path for summary jsonl file saved during training."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path for saved model bin file."
    )
    parser.add_argument(
        "--test_file", type=str, default="data/test_audiocaps_subset.json",
        help="json file containing the test prompts for generation."
    )
    parser.add_argument(
        "--text_key", type=str, default="captions",
        help="Key containing the text in the json file."
    )
    parser.add_argument(
        "--test_references", type=str, default="data/audiocaps_test_references/subset",
        help="Folder containing the test reference wav files."
    )
    parser.add_argument(
        "--num_steps", type=int, default=200,
        help="How many denoising steps for generation.",
    )
    parser.add_argument(
        "--guidance", type=float, default=3,
        help="Guidance scale for classifier free guidance."
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1,
        help="How many samples per prompt.",
    )
    parser.add_argument(
        "--num_test_instances", type=int, default=-1,
        help="How many test instances to evaluate.",
    )
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    
    train_args = dotdict(json.loads(open(args.original_args).readlines()[0]))
    print(train_args)
    if "hf_model" not in train_args:
        train_args["hf_model"] = None
    
    # Load Models #
    if train_args.hf_model:
        tango = Tango(train_args.hf_model, "cpu")
        vae, stft, model = tango.vae.cuda(), tango.stft.cuda(), tango.model.cuda()
    else:
        name = "audioldm-s-full-v2"
        # name = "models_watkins_fixed"
        vae, stft = build_pretrained_models(name)
        
        model = AudioDiffusion(
            train_args.text_encoder_name, train_args.scheduler_name, train_args.unet_model_name, train_args.unet_model_config, train_args.snr_gamma
        )
        if USE_CUDA:
            vae, stft = vae.cuda(), stft.cuda()
            model = model.cuda()

        audioldm = AudioLDMPipeline.from_pretrained(f"cvssp/{name}")
        
        if USE_CUDA:
            audioldm.vae.cuda()
            audioldm.vocoder.cuda()

            if MATCH_AUDIOLDM:
                model.text_encoder = audioldm.text_encoder.cuda()
                model.unet = audioldm.unet.cuda()
                
                model.tokenizer = audioldm.tokenizer
            if USE_AUDIOLDM_VAE:
                vae = audioldm.vae.cuda()
        else:
            if MATCH_AUDIOLDM:
                model.text_encoder = audioldm.text_encoder
                model.unet = audioldm.unet
                model.tokenizer = audioldm.tokenizer
            if USE_AUDIOLDM_VAE:
                vae = audioldm.vae

        # model.vae = audioldm.vae
        # model.eval()
    
    # Load Trained Weight #
    if USE_AUDIOLDM_VAE:
        device = vae.device
    else:
        device = vae.device()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    # model.unet.save_pretrained

    if MATCH_AUDIOLDM:
        audioldm.unet = model.unet
        audioldm.text_encoder = model.text_encoder

        audioldm_save_path = "audioldm_" + "_".join(args.model.split("/")[1:-1])
        print("saving audioldm to", audioldm_save_path)
        audioldm.save_pretrained(audioldm_save_path)
    
    scheduler = audioldm.scheduler

    evaluator = EvaluationHelper(16000, "cuda:0")
    
    if args.num_samples > 1:
        clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
        clap.eval()
        clap_processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
    
    wandb.init(project="Text to Audio Diffusion Evaluation")

    def audio_text_matching(waveforms, text, sample_freq=16000, max_len_in_seconds=10):
        new_freq = 48000
        resampled = []
        
        for wav in waveforms:
            x = torchaudio.functional.resample(torch.tensor(wav, dtype=torch.float).reshape(1, -1), orig_freq=sample_freq, new_freq=new_freq)[0].numpy()
            resampled.append(x[:new_freq*max_len_in_seconds])

        inputs = clap_processor(text=text, audios=resampled, return_tensors="pt", padding=True, sampling_rate=48000)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = clap(**inputs)

        logits_per_audio = outputs.logits_per_audio
        ranks = torch.argsort(logits_per_audio.flatten(), descending=True).cpu().numpy()
        return ranks
    
    # Load Data #
    if train_args.prefix:
        prefix = train_args.prefix
    else:
        prefix = ""
        
    # text_prompts = [json.loads(line)[args.text_key] for line in open(args.test_file).readlines()]
    # text_prompts = [prefix + inp for inp in text_prompts]
    # "Spinner Dolphin call", "Beluga, White Whale call", "Gunshot", 
    text_prompts = ["Beluga, White Whale call", "Spinner Dolphin call", "Clymene Dolphin call", "Bearded Seal call", "Minke Whale call", "Rooster", "Toilet flush", "Buff-bellied Pipit", "American Goldfinch", "Grass coqui"]
    
    if args.num_test_instances != - 1:
        text_prompts = text_prompts[:args.num_test_instances]
    
    # Generate #
    num_steps, guidance, batch_size, num_samples = args.num_steps, args.guidance, args.batch_size, args.num_samples
    all_outputs = []
    
    generator = torch.Generator(device=device)
    for k in tqdm(range(0, len(text_prompts), batch_size)):
        text = text_prompts[k: k+batch_size]
        print("text", text)
        
        with torch.no_grad():
            print("num sam", num_samples)

            latents = model.inference(text, scheduler, num_steps, guidance, num_samples, disable_progress=True, generator=generator)
            print("latents shape", latents.shape)
            if USE_AUDIOLDM_VAE:
                mel = audioldm.decode_latents(latents)
                wave = audioldm.mel_spectrogram_to_waveform(mel)
            else:
                mel = vae.decode_first_stage(latents)
                wave = vae.decode_to_waveform(mel)
                print("WAVSHAPE", wave.shape)
            all_outputs += [item for item in wave]
        
        # waves2 = audioldm(prompt=text, generator=generator).audios

        # for index, w in enumerate(waves2):
        #     print("wave", w)
        #     scipy.io.wavfile.write(f"outputs/output_{index}.wav", rate=16000, data=w)

            
    # Save #
    exp_id = str(int(time.time()))
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    
    if num_samples == 1:
        output_dir = "outputs/{}_{}_steps_{}_guidance_{}".format(exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance)
        os.makedirs(output_dir, exist_ok=True)
        for j, wav in enumerate(all_outputs):
            sf.write("{}/output_{}.wav".format(output_dir, j), wav, samplerate=16000)

        result = evaluator.main(output_dir, args.test_references)
        result["Steps"] = num_steps
        result["Guidance Scale"] = guidance
        result["Test Instances"] = len(text_prompts)
        wandb.log(result)
        
        result["scheduler_config"] = dict(scheduler.config)
        result["args"] = dict(vars(args))
        result["output_dir"] = output_dir

        with open("outputs/summary.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n\n")
            
    else:
        for i in range(num_samples):
            output_dir = "outputs/{}_{}_steps_{}_guidance_{}/rank_{}".format(exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance, i+1)
            os.makedirs(output_dir, exist_ok=True)
        
        groups = list(chunks(all_outputs, num_samples))
        for k in tqdm(range(len(groups))):
            wavs_for_text = groups[k]
            rank = audio_text_matching(wavs_for_text, text_prompts[k])
            ranked_wavs_for_text = [wavs_for_text[r] for r in rank]
            
            for i, wav in enumerate(ranked_wavs_for_text):
                output_dir = "outputs/{}_{}_steps_{}_guidance_{}/rank_{}".format(exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance, i+1)
                sf.write("{}/output_{}.wav".format(output_dir, k), wav, samplerate=16000)
            
        # Compute results for each rank #
        for i in range(num_samples):
            output_dir = "outputs/{}_{}_steps_{}_guidance_{}/rank_{}".format(exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance, i+1)
            result = evaluator.main(output_dir, args.test_references)
            result["Steps"] = num_steps
            result["Guidance Scale"] = guidance
            result["Instances"] = len(text_prompts)
            result["clap_rank"] = i+1
            
            wb_result = copy.deepcopy(result)
            wb_result = {"{}_rank{}".format(k, i+1): v for k, v in wb_result.items()}
            wandb.log(wb_result)
            
            result["scheduler_config"] = dict(scheduler.config)
            result["args"] = dict(vars(args))
            result["output_dir"] = output_dir

            with open("outputs/summary.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n\n")
        
if __name__ == "__main__":
    main()