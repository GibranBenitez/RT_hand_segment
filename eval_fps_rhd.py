import time
import torch
import torch.backends.cudnn as cudnn
import os
from ptsemseg.utils import convert_state_dict
from argparse import ArgumentParser
from ptflops import get_model_complexity_info
import cpuinfo
# from ptsemseg.models.FASSDNet import FASSDNet
# from ptsemseg.models.FASSDNetL1 import FASSDNet
# from ptsemseg.models.FASSDNetL2 import FASSDNet

def compute_speed(args, model, input_size, device, iteration=1000):
    model.eval()
    if device == "cpu":
        device = torch.device("cpu")
        model = model.to(device)
    else:
        cudnn.benchmark = True
        device = torch.device(int(device))
        model = model.to(device)
    
    input = torch.randn(*input_size, device=device)
    
    for _ in range(50):
        model(input)

    print('\n============Starting===========')
    print('Model: {}, @{}x{} on: {}'.format(args.weights.split("/")[-1],input_size[2],input_size[3], device))
    flops, params = get_model_complexity_info(model, (input_size[1],input_size[2],input_size[3]), as_strings=True, print_per_layer_stat=False)
    print('Flops:  ' + flops)
    print('Params: ' + params)
    print('=========Speed Testing=========')
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iteration):
        model(input)
        print("iteration {}/{}".format(_, iteration), end='\r')
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    speed_time = elapsed_time / iteration * 1000
    fps = iteration / elapsed_time

    print('Elapsed Time: [%.2f s / %d iter]' % (elapsed_time, iteration))
    print('Speed Time: %.2f ms / iter   FPS: %.2f' % (speed_time, fps))
    return speed_time, fps


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--size", type=str, default="480,640", help="input size of model")
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--classes', type=int, default=14)
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument("--gpus", type=str, default="cpu", help="gpu ids (default: 0)")
    parser.add_argument("--alpha", default=2, nargs="?", type=int)
    parser.add_argument("--model", nargs="?", type=str,
        default="DDRNet",# FASSDNet, HardNet, DABNet, ENet, FastSCNN, DDRNet
        help="Model variation to use",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default ='weights_hand/DDRNet_finger_ipn.pkl',
        help="location of the weights",
    )
    args = parser.parse_args()

    # Print PC stats
    print(cpuinfo.get_cpu_info()['brand_raw'])
    print(torch.cuda.get_device_name(0))

    if args.model == "FASSDNet":                        # 1xTitanRTX (gp33)
        from ptsemseg.models.FASSDNet import FASSDNet as SegModel
        args.weights = "weights_hand/FASSDNet_finger_ipn.pkl" 
    elif args.model == "HardNet":                       # 1xTitanRTX (gp35)
        from ptsemseg.models.hardnet import hardnet as SegModel
        args.weights = "weights_hand/HardNet_finger_ipn.pkl" 
    elif args.model == "DABNet":                        # 2xTitanRTX (gp36)
        from ptsemseg.models.DABNet import DABNet as SegModel
        args.weights = "weights_hand/DABNet_finger_ipn.pkl" 
    elif args.model == "FastSCNN":                      # 2x2080Ti (gp40) | 1xTitanRTX
        from ptsemseg.models.FastSCNN import FastSCNN as SegModel
        args.weights = "weights_hand/FastSCNN_finger_ipn.pkl" 
    elif args.model == "DDRNet":                        # 2x2080Ti (gp30) | 1xTitanRTX
        from ptsemseg.models.ddrnet23_slim import DualResNet as SegModel
        args.weights = "weights_hand/DDRNet_finger_ipn.pkl" 
    else:
        print("No valid model found")


    h, w = map(int, args.size.split(','))            

    model = SegModel(n_classes=args.classes, alpha=args.alpha).cuda()
    state = convert_state_dict(torch.load(args.weights)["model_state"])
    model.load_state_dict(state)
        
    compute_speed(args, model, (args.batch_size, args.num_channels, h, w), args.gpus, iteration=args.iter)
