import time
import torch
import torch.backends.cudnn as cudnn
import os
from ptsemseg.utils import convert_state_dict
from argparse import ArgumentParser
from torch2trt import TRTModule
from torch2trt import torch2trt
# from ptsemseg.models.FASSDNet import FASSDNet
# from ptsemseg.models.FASSDNetL1 import FASSDNet
# from ptsemseg.models.FASSDNetL2 import FASSDNet

def compute_speed(args, model, input_size, device, iteration=1000):
    torch.cuda.set_device(device)
    cudnn.benchmark = True

    model.eval()
    model = model.cuda()
    
    input = torch.randn(*input_size, device=device)
    
    for _ in range(50):
        model(input)

    print('\n============Starting===========')
    print('Model: {}, @{}x{}'.format(args.weights.split("/")[-1],input_size[2],input_size[3]))
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
    parser.add_argument("--gpus", type=str, default="0", help="gpu ids (default: 0)")
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

    model_path = "./weights_hand/" + args.weights.split("/")[-1][:-4] + "_trt_" + str(h) + "x" + str(w) + "_fp16.pth"    
    if os.path.isfile(model_path):
        print("Loading weights from {}".format(model_path))
        model = TRTModule()
        model.load_state_dict(torch.load(model_path))
    else:        
        print("No existing model found. Converting and saving TRT model...")
        model_path_non_trt = args.weights
        model = SegModel(n_classes=args.classes, alpha=args.alpha).cuda()
        state = convert_state_dict(torch.load(model_path_non_trt)["model_state"])    
        model.load_state_dict(state, strict=False)                
        model.eval()
            
        x = torch.rand((1, 3, int(h), int(w))).cuda()
        model = torch2trt(model, [x], fp16_mode=True)                
        torch.save(model.state_dict(), model_path)
        print("Done!")

    args.weights = model_path
        
    compute_speed(args, model, (args.batch_size, args.num_channels, h, w), int(args.gpus), iteration=args.iter)