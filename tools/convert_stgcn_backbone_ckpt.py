# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Remove ST-GCN classification head weights from a '
        'checkpoint.')
    parser.add_argument('input', help='Input checkpoint path')
    parser.add_argument('output', help='Output checkpoint path')
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.input, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    filtered_state_dict = {
        k: v
        for k, v in state_dict.items() if not k.startswith('cls_head.')
    }

    if 'state_dict' in checkpoint:
        checkpoint['state_dict'] = filtered_state_dict
        output = checkpoint
    else:
        output = filtered_state_dict

    torch.save(output, args.output)
    print(f'Saved backbone checkpoint to {args.output}')


if __name__ == '__main__':
    main()
