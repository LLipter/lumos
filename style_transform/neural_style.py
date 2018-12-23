import os
import sys
import time
import re
import cv2

import numpy as np
import torch
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

import utils
from transformer_net import TransformerNet
from vgg import Vgg16

def video2imgs(videoname, outputvideo):
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    img_list = []
    transform_list = []
    cap = cv2.VideoCapture(videoname)
    width = cap.get(3)
    height = cap.get(4)
    intwidth = int(width)
    intheight = int(height)
    print(width)
    print(height)
    Vwriter = cv2.VideoWriter(outputvideo, fourcc, 25, (intwidth, intheight), True)
    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret and counter < 25:
            counter = counter + 1
            img_list.append(frame)
            print("frame added", counter)
        else:
            break
    modelsrc = "/Users/anneyino/Desktop/models/candy.pth"        
    transform_list = stylize(img_list, 0, modelsrc)
    cap.release()
    print("style transfromed")
    for transimg in transform_list:
        img = transimg.clone().clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0).astype("uint8")
        img = Image.fromarray(img)
        img = np.array(img)
        img = img[..., [2, 1, 0]]
        Vwriter.write(img)



def imgs2video(videoname, imglist):
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    Vwriter = cv2.VideoWriter(videoname, fourcc, 25, (1000, 912), True)
    for img in imglist:
        Vwriter.write(img)

def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()

    # save model
    transformer.eval().cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(content_image, has_cuda, model):
    img_list = []
    device = torch.device("cuda" if has_cuda else "cpu")
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    for i in range(len(content_image)):
        content_image[i] = content_transform(content_image[i])
        content_image[i] = content_image[i].unsqueeze(0).to(device)
    with torch.no_grad():
            style_model = TransformerNet()
            state_dict = torch.load(model)
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            for j in range(len(content_image)):
                output = style_model(content_image[j]).cpu()
                img_list.append(output[0])
                print("transformed")
                #utils.save_image(outputimg[j], output[0])
    return img_list

def stylize_onnx_caffe2(content_image, args):
    """
    Read ONNX model and run it using Caffe2
    """

    assert not args.export_onnx

    import onnx
    import onnx_caffe2.backend

    model = onnx.load(args.model)

    prepared_backend = onnx_caffe2.backend.prepare(model, device='CUDA' if args.cuda else 'CPU')
    inp = {model.graph.input[0].name: content_image.numpy()}
    c2_out = prepared_backend.run(inp)[0]

    return torch.from_numpy(c2_out)


def main():
    img01 = utils.load_image("/Users/anneyino/Desktop/models/img01.jpg")
    #img01 = img01.resize((2160,1080))
    img02 = cv2.imread("/Users/anneyino/Desktop/models/img01.jpg")
    print(img02[4])
    img01 = np.array(img01)
    #print(img01[2])
    img03 = img01[...,[2, 1, 0]]
    print(img03[4])

    print(img02.shape)
    print(img03.shape)

    #img02 = cv2.resize(img02,(2160,1080),interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("/Users/anneyino/Desktop/testimage.jpg",img03,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
    vdname = "/Users/anneyino/Desktop/models/testvideo.mp4"
    outputname = "/Users/anneyino/Desktop/test.mp4"
    #imagelist = video2imgs(vdname)
    img_list = []
    #cap = cv2.VideoCapture(vdname)
    counter = 0
    #while cap.isOpened():
        #ret, frame = cap.read()
        #if ret and counter < 50:
            #counter = counter + 1
            #img_list.append(frame)
        #else:
            #break
    #cap.release()
    video2imgs(vdname, outputname)


if __name__ == "__main__":
    main()
