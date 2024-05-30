import os
import yaml
import argparse
import torch
import transformers


from chatcaptioner.chat import set_openai_key, caption_images, get_instructions
from chatcaptioner.blip2 import Blip2
from chatcaptioner.utils import RandomSampledDataset, plot_img, print_info
from X2_VLM.CBIR import create_dataset
from X2_VLM.models.model_generation import XVLMForVQA


def parse():
    parser = argparse.ArgumentParser(description='Generating captions in test datasets.')
    parser.add_argument('--data_root', type=str, default='/run/media/filippo/Seagate Basic/DeepFashion2/',
                        help='root path to the datasets')
    parser.add_argument('--save_root', type=str, default='/home/filippo/PycharmProjects/caption-guided-cbir/data/chatcaptioner/',
                        help='root path for saving results')
    parser.add_argument('--exp_tag', type=str, required=True, 
                        help='tag for this experiment. caption results will be saved in save_root/exp_tag')
    parser.add_argument('--datasets', nargs='+', choices=['artemis', 'cc_val', 'coco_val', 'para_test', 'pascal', 'deepfashion2'], default=['deepfashion2'],
                        help='Names of the datasets to use in the experiment. Valid datasets include artemis, cc_val, coco_val. Default is coco_val')
    parser.add_argument('--n_rounds', type=int, default=10, 
                        help='Number of QA rounds between GPT3 and BLIP-2. Default is 10, which costs about 2k tokens in GPT3 API.')
    parser.add_argument('--n_context', type=int, default=1,
                        help='Number of QA rounds visible to BLIP-2. Default is 1, which means BLIP-2 only remember one previous question. -1 means BLIP-2 can see all the QA rounds')
    parser.add_argument('--model', type=str, default='microsoft/Phi-3-mini-128k-instruct', choices=['microsoft/Phi-3-mini-128k-instruct'],
                        help='model used to ask question. can be gpt3, chatgpt, or its concrete tags in openai system')
    parser.add_argument('--device_id', type=int, default=0, 
                        help='Which GPU to use.')
    parser.add_argument('--img_root', type=str, required=True,
                        help='The path to the folder containing the images.')
    parser.add_argument('--config', type=str, required=True,
                        help='The path to the folder containing the images.')

    args = parser.parse_args()
    return args

    
def main(args):
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    question_model = args.model
    vqa_model = XVLMForVQA(config)
    for dataset_name in args.datasets:
        # load the dataset
        train_dataset, val_dataset, test_dataset = create_dataset('cbir', config)
        # preparing the folder to save results
        save_path = os.path.join(args.save_root, args.exp_tag, dataset_name)
        if not os.path.exists(save_path):
            os.makedirs(os.path.join(save_path, 'caption_result'))
        with open(os.path.join(save_path, 'instruction.yaml'), 'w') as f:
            yaml.dump(get_instructions(), f)

        # start caption
        caption_images(vqa_model,
                       train_dataset,
                       save_path=save_path,
                       n_rounds=args.n_rounds,
                       n_context=args.n_context,
                       model=question_model,
                       print_mode='no')
    

if __name__ == '__main__':
    args = parse()
    main(args)