import tensorflow as tf
import os
import json
import subprocess
from scipy.misc import imread, imresize
from scipy import misc
import timeit
from train import build_forward
from utils.annolist import AnnotationLib as al
from utils.train_utils import add_rectangles, rescale_boxes

import cv2
import argparse

def get_image_dir(args):
    # weights_iteration = int(args.weights.split('-')[-1])
    # expname = '_' + args.expname if args.expname else ''
    # image_dir = '%s/images_%s_%d%s' % (os.path.dirname(args.weights), os.path.basename(args.image)[:-5], weights_iteration, expname)
    image_dir = args.output_dir
    return image_dir

def get_results(args, H):
    tf.reset_default_graph()
    H["grid_width"] = H["image_width"] / H["region_size"]
    H["grid_height"] = H["image_height"] / H["region_size"]
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
    if H['use_rezoom']:
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
        grid_area = H['grid_height'] * H['grid_width']
        pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], 2])), [grid_area, H['rnn_len'], 2])
        if H['reregress']:
            pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.weights)

        pred_annolist = al.AnnoList()

        # true_annolist = al.parse(args.test_boxes)
        data_dir = os.path.dirname(args.image)
        image_dir = get_image_dir(args)
        subprocess.call('mkdir -p %s' % image_dir, shell=True)
        # for i in range(len(true_annolist)):
            # true_anno = true_annolist[i]
        orig_img = imread('%s/%s' % (data_dir, args.image.split("/")[-1]))[:,:,:3]
        img = imresize(orig_img, (H["image_height"], H["image_width"]), interp='cubic')
        feed = {x_in: img}
        (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
        pred_anno = al.Annotation()
        pred_anno.imageName = args.image
        new_img, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
                                            use_stitching=True, rnn_len=H['rnn_len'], min_conf=args.min_conf, tau=args.tau, show_suppressed=args.show_suppressed)
        
        pred_anno.rects = rects
        imname = '%s/%s' % (args.output_dir, os.path.basename(args.image))
        pred_anno.imagePath = os.path.abspath(imname)
        pred_anno = rescale_boxes((H["image_height"], H["image_width"]), pred_anno, orig_img.shape[0], orig_img.shape[1])
        pred_annolist.append(pred_anno)
            
        misc.imsave(imname, new_img)

    return pred_annolist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--expname', default='')
    parser.add_argument('--image', required=True)
    parser.add_argument('--output_dir', required=False, default="predictions")
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--logdir', default='output')
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--tau', default=0.25, type=float)
    parser.add_argument('--min_conf', default=0.2, type=float)
    parser.add_argument('--show_suppressed', default=False, type=bool)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    hypes_file = '%s/hypes.json' % os.path.dirname(args.weights)
    with open(hypes_file, 'r') as f:
        H = json.load(f)
    # expname = args.expname + '_' if args.expname else ''
    pred_boxes = '%s/%s' % (args.output_dir, 'predicted_boxes.json')
    #true_boxes = '%s.gt_%s%s' % (args.weights, expname, os.path.basename(args.test_boxes))

    t0 = timeit.default_timer()
    pred_annolist = get_results(args, H)
    print timeit.default_timer() - t0
    pred_annolist.save(pred_boxes)
    print(pred_boxes)
    #true_annolist.save(true_boxes)
    #print(true_boxes)


if __name__ == '__main__':
    main()
