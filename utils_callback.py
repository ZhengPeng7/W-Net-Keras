import os
import cv2
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
import matplotlib.pyplot as plt


def callbacks_during_train(model, dis_x, dis_y, dis_path, net, epoch):
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    # show prediction
    pred = np.squeeze(model.predict(np.expand_dims(dis_x, axis=0)))
    _, (ax_x_ori, ax_y, ax_pred) = plt.subplots(1, 3, figsize=(20, 4))
    ax_x_ori.imshow(cv2.cvtColor(cv2.imread(dis_path), cv2.COLOR_BGR2RGB))
    ax_x_ori.set_title('Original Image')
    ax_y.imshow(np.squeeze(dis_y), cmap=plt.cm.jet)
    ax_y.set_title('Ground_truth: ' + str(np.sum(dis_y)))
    ax_pred.imshow(pred, cmap=plt.cm.jet)
    ax_pred.set_title('Prediction: ' + str(np.sum(pred)))
    plt.savefig('tmp/{}_{}-epoch.png'.format(net, epoch))
    return None


def eval_loss(model, x, y, quality=False):
    preds, DM, GT = [], [], []
    losses_SFN, losses_MAE, losses_MAPE, losses_MSE = [], [], [], []
    for idx_pd in range(x.shape[0]):
        pred = model.predict(x[idx_pd:idx_pd+1])
        preds.append(np.squeeze(pred))
        DM.append(np.squeeze(y[idx_pd:idx_pd+1]))
        GT.append(round(np.sum(y[idx_pd:idx_pd+1])))    # To make sure the GT is an integral value
    for idx_pd in range(len(preds)):
        losses_SFN.append(np.mean(np.square(preds[idx_pd] - DM[idx_pd])))     # mean of Frobenius norm
        losses_MAE.append(np.abs(np.sum(preds[idx_pd]) - GT[idx_pd]))
        losses_MAPE.append(np.abs(np.sum(preds[idx_pd]) - GT[idx_pd]) / GT[idx_pd])
        losses_MSE.append(np.square(np.sum(preds[idx_pd]) - GT[idx_pd]))

    loss_SFN = np.mean(losses_SFN)
    loss_MAE = np.mean(losses_MAE)
    loss_MAPE = np.mean(losses_MAPE)
    loss_MSE = np.sqrt(np.mean(losses_MSE))
    if quality:
        psnr, ssim = [], []
        for idx_pd in range(len(preds)):
            data_range = max([np.max(preds[idx_pd]), np.max(DM[idx_pd])]) - min([np.min(preds[idx_pd]), np.min(DM[idx_pd])])
            psnr_ = compare_psnr(preds[idx_pd], DM[idx_pd], data_range=data_range)
            ssim_ = compare_ssim(preds[idx_pd], DM[idx_pd], data_range=data_range)
            psnr.append(psnr_)
            ssim.append(ssim_)
        return loss_MAE, loss_MSE, loss_SFN, loss_MAPE, np.mean(psnr), np.mean(ssim)
    return loss_MAE, loss_MSE, loss_SFN, loss_MAPE
