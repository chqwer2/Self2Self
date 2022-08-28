import tensorflow as tf
import network.Punet
import numpy as np

import util
import cv2
import os
import time, math

TF_DATA_TYPE = tf.float32
LEARNING_RATE = 1e-4
N_PREDICTION = 100
N_SAVE = 1000
N_STEP = 150000
pyramid=False

mask_rate = 0.7

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


usm = False

def train(file_path, dropout_rate, sigma=25, is_realnoisy=False):
    print(file_path)
    tf.reset_default_graph()
    gt = util.load_np_image(file_path)
    _, w, h, c = np.shape(gt)
    
    model_name = 'Self2Self_cycle'
    if pyramid:
        model_name += '_pyramid'
        
    if usm:
        model_name + '_usm'
    model_path = file_path[0:file_path.rfind(".")] + "/" + str(sigma) + f"/model/{model_name}/"
    
    
    os.makedirs(model_path, exist_ok=True)
    noisy_input = util.add_gaussian_noise(gt, model_path, sigma)
    #print("input:", noisy)
    model = network.Punet.build_denoising_unet(noisy_input, 1 - dropout_rate, mask_rate=mask_rate, is_realnoisy, pyramid=pyramid)

    loss = model['training_error']
    summay = model['summary']
    saver = model['saver']
    our_image = model['our_image']
    is_flip_lr = model['is_flip_lr']
    is_flip_ud = model['is_flip_ud']
    noisy = model['noisy']
    avg_op = model['avg_op']
    slice_avg = model['slice_avg']
    
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    start = end = time.time()
    avg_loss = 0
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(model_path, sess.graph)
        for step in range(N_STEP):
            feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2), noisy:noisy_input}
            
            _, _op, loss_value, merged, o_image = sess.run([optimizer, avg_op, loss, summay, our_image],
                                                           feed_dict=feet_dict)
            avg_loss += loss_value
            
            if (step + 1) % 20 == 0:
                print("Elapsed %s  iteration %d, loss = %.4f" % (timeSince(start, float(step + 1)/N_STEP), step + 1, avg_loss / N_SAVE))
            
            
            if (step + 1) % N_SAVE == 0:

                # print("After %d training step(s)" % (step + 1),
                #       "loss  is {:.9f}".format(avg_loss / N_SAVE))
                
                print("model path:", model_path)
                avg_loss = 0
                sum = np.float32(np.zeros(our_image.shape.as_list()))
                for j in range(N_PREDICTION):  # N_PREDICTION bigger
                    feet_dict = {is_flip_lr: np.random.randint(0, 2), is_flip_ud: np.random.randint(0, 2), noisy:noisy_input}
                    o_avg, o_image = sess.run([slice_avg, our_image], feed_dict=feet_dict)
                    sum += o_image
                o_image = np.squeeze(np.uint8(np.clip(sum / N_PREDICTION, 0, 1) * 255))
                o_avg = np.squeeze(np.uint8(np.clip(o_avg, 0, 1) * 255))
                
                if is_realnoisy:
                    cv2.imwrite(model_path + 'Self2Self-' + str(step + 1) + '.png', o_avg)
                else:
                    cv2.imwrite(model_path + 'Self2Self-' + str(step + 1) + '.png', o_image)
                saver.save(sess, model_path + "model.ckpt-" + str(step + 1))
                
                o_avg_BGR = cv2.cvtColor(o_avg, cv2.COLOR_RGB2BGR)
                blur_img = cv2.GaussianBlur(o_avg_BGR, (0, 0), 5)
                alpha = 1.3
                usm_avg_BGR = cv2.addWeighted(o_avg_BGR, alpha, blur_img, 1-alpha, 0)
                o_avg_usm = np.clip(cv2.cvtColor(o_avg_BGR, cv2.COLOR_RGB2BGR), 0, 255)
                cv2.imwrite(model_path + 'Self2Self-' + str(step + 1) + '-usm_1.3.png', o_avg_usm)
                
            if (step + 1) % 25000 == 0:  
                if usm:
                    o_avg_BGR = cv2.cvtColor(o_avg, cv2.COLOR_RGB2BGR)
                    blur_img = cv2.GaussianBlur(o_avg_BGR, (0, 0), 5)
                    alpha = 1.3
                    usm_avg_BGR = cv2.addWeighted(o_avg_BGR, alpha, blur_img, 1-alpha, 0)
                    o_avg = np.clip(cv2.cvtColor(o_avg_BGR, cv2.COLOR_RGB2BGR), 0, 255)
                
                noisy_input = noisy_input*0.8 + o_avg*0.2/255
                cv2.imwrite(model_path + 'Input-' + str(step + 1) + '.png', np.squeeze(np.uint8(noisy_input*255)))
                print("substitue")
                
            summary_writer.add_summary(merged, step)


if __name__ == '__main__':
#     path = './testsets/Set9/'
#     file_list = os.listdir(path)
#     for sigma in [25, 50, 75, 100]:
#         for file_name in file_list:
#             if not os.path.isdir(path + file_name):
#                 train(path + file_name, 0.3, sigma)


    path = './testsets/MyBSD_noise50/'
    file_list = os.listdir(path)
    sigma = -1
    for file_name in file_list:
        if not os.path.isdir(path + file_name):
            train(path + file_name, 0.3, sigma, is_realnoisy = True)

    # # path = './testsets/PolyU/'
    # path = './testsets/BSD68/0018.png'
    # # file_list = os.listdir(path)
    # sigma = -1
    # # for file_name in file_list:
    # #     if not os.path.isdir(path + file_name):
    # train(path , 0.3, sigma, is_realnoisy = True)
