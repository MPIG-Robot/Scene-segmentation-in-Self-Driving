import os
import torch
import numpy as np
import scipy.misc as misc


from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import convert_state_dict

try:
    import pydensecrf.densecrf as dcrf
except:
    print(
        "Failed to import pydensecrf,\
           CRF post-processing will not work"
    )


def test(img_path,model_path,dataset,out_path):
    img_norm = True
    dcrf = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_file_name = os.path.split(model_path)[1]
    model_name = model_file_name[: model_file_name.find("_")]

    # Setup image
    print("Read Input Image from : {}".format(img_path))
    img = misc.imread(img_path)

    data_loader = get_loader(dataset)
    loader = data_loader(root=None, is_transform=True, img_norm=True, test_mode=True)
    n_classes = loader.n_classes

    resized_img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]), interp="bicubic")

    orig_size = img.shape[:-1]
    if model_name in ["pspnet", "icnet", "icnetBN"]:
        # uint8 with RGB mode, resize width and height which are odd numbers
        img = misc.imresize(img, (orig_size[0] // 2 * 2 + 1, orig_size[1] // 2 * 2 + 1))
    else:
        img = misc.imresize(img, (loader.img_size[0], loader.img_size[1]))

    img = img[:, :, ::-1]
    img = img.astype(np.float64)
    img -= loader.mean
    if img_norm:
        img = img.astype(float) / 255.0

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    # Setup Model
    model_dict = {"arch": model_name}
    model = get_model(model_dict, n_classes, version=dataset)
    state = convert_state_dict(torch.load(model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    images = img.to(device)
    outputs = model(images)

    if dcrf:
        unary = outputs.data.cpu().numpy()
        unary = np.squeeze(unary, 0)
        unary = -np.log(unary)
        unary = unary.transpose(2, 1, 0)
        w, h, c = unary.shape
        unary = unary.transpose(2, 0, 1).reshape(loader.n_classes, -1)
        unary = np.ascontiguousarray(unary)

        resized_img = np.ascontiguousarray(resized_img)

        d = dcrf.DenseCRF2D(w, h, loader.n_classes)
        d.setUnaryEnergy(unary)
        d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=resized_img, compat=1)

        q = d.inference(50)
        mask = np.argmax(q, axis=0).reshape(w, h).transpose(1, 0)
        decoded_crf = loader.decode_segmap(np.array(mask, dtype=np.uint8))
        dcrf_path = out_path[:-4] + "_drf.png"
        misc.imsave(dcrf_path, decoded_crf)
        print("Dense CRF Processed Mask Saved at: {}".format(dcrf_path))

    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    if model_name in ["pspnet", "icnet", "icnetBN"]:
        pred = pred.astype(np.float32)
        # float32 with F mode, resize back to orig_size
        pred = misc.imresize(pred, orig_size, "nearest", mode="F")

    decoded = loader.decode_segmap(pred)
    print("Classes found: ", np.unique(pred))
    misc.imsave(out_path, decoded)
    print("Segmentation Mask Saved at: {}".format(out_path))


if __name__ == "__main__":

    model_path = "./runs/your_own_path/fcn8s_camvid_best_model.pkl"
    dataset = "camvid"
    for i in range(100):
        img_num = str(i*30+930).zfill(5)
        img_path = "/home/lq/data/CamVid/train/0006R0_f"+str(img_num) +".png"
        out_path = "./result/"+str(i)+".png"
        print("img_path: ",img_path)
        print("out_path: ",out_path)
        test(img_path,model_path,dataset,out_path)
	
    
