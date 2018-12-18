import os
import glob

import click
import cv2
import numpy as np


def read_binary(fid, length, dtype):
    """
    Read binary file

    :param fid: file object
    :param length: data length to read
    :type length: int
    :param dtype: data type
    :return: desired binary data
    :rtype: numpy.ndarray
    """
    if dtype is np.str:
        dt = np.uint8  # WARNING: assuming 8-bit ASCII for np.str!
    else:
        dt = dtype

    data_array = np.fromfile(fid, dt, length)
    return data_array


def decode_yuv(data, height, width):
    """
    Decode YUV image from binary data

    :param data: binary data
    :type data: numpy.ndarray
    :param height: image height
    :type height: int
    :param width: image width
    :type width: int
    :return: YUV image
    :rtype: tuple(numpy.ndarray)
    """
    # construct YUV from binary data
    y = data[:height * width].reshape(height, width, order='F')
    u = data[height * width:int(height * width * 1.5) - 1:2].reshape(height // 2, width // 2, order='F')
    v = data[int(height * width * 1.5) - 1:height * width * 2 - 2:2].reshape(height // 2, width // 2, order='F')

    # enlarge U and V channel
    enlarge_u = cv2.resize(u, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
    enlarge_v = cv2.resize(v, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)

    return y, enlarge_u, enlarge_v


def yuv2rgb(y, u, v, use_cv2=True):
    """
    Convert YUV420 (NV12) image into RGB color space

    :param y: image data of Y channel
    :type y: numpy.ndarray
    :param u: image data of U channel
    :type u: numpy.ndarray
    :param v: image data of V channel
    :type v: numpy.ndarray
    :param use_cv2: whether or not use existing method in opencv to convert color space
    :type use_cv2: bool
    :return: image in RGB color space
    :rtype: numpy.ndarray
    """
    img_yuv = cv2.merge([y, u, v])

    if use_cv2:
        img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    else:
        r = y + 1.140 * (v - 128.)
        g = y - 0.581 * (v - 128.) - 0.395 * (u - 128.)
        b = y + 2.032 * (u - 128.)
        img_rgb = np.clip(cv2.merge([r, g, b]), 0, 255)

    return img_rgb


def bin2img(bin_file, out_dir, height=640, width=480):
    """
    Convert binary image file into RGB png image

    :param bin_file: path to binary file
    :type bin_file: str
    :param out_dir: path to save result image
    :type out_dir: str
    :param height: image height
    :type height: int
    :param width: image width
    :type width: int
    """
    # check output directory
    os.makedirs(out_dir, exist_ok=True)
    # length for image binary
    data_len = height * width * 2 - 2
    # read image data
    image_idx = 0
    with open(bin_file, 'rb') as f:
        while True:
            # read binary
            print('Processing frame {}'.format(image_idx))
            yuv_raw = read_binary(f, data_len, np.uint8)
            if len(yuv_raw) < data_len:
                break
            # convert yuv to rgb
            y, u, v = decode_yuv(yuv_raw, height, width)
            img = yuv2rgb(y, u, v)
            img = np.flipud(img)
            # save image
            out_path = os.path.join(out_dir, '{:05d}.png'.format(image_idx))
            cv2.imwrite(out_path, img[..., ::-1])
            # increase index
            image_idx += 1
    print('Done')


@click.command(help='Decode binary file into RGB images')
@click.argument('bin-file', type=click.Path(exists=True))
@click.option('--out-dir', type=click.Path(exists=True))
@click.option('--height', type=int, default=640)
@click.option('--width', type=int, default=480)
def decode_bin_file(bin_file, out_dir, height, width):
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(bin_file), 'png')
    bin2img(bin_file, out_dir, height, width)


@click.command(help='Decode binary files into RGB images')
@click.argument('data-root', type=click.Path(exists=True))
@click.option('--out-dir', type=click.Path(exists=True))
@click.option('--height', type=int, default=640)
@click.option('--width', type=int, default=480)
def decode_bin_file(data_root, out_dir, height, width):
    bin_files = glob.iglob(os.path.join(data_root, '**', '*.bin'), recursive=True)
    for bin_file in bin_files:
        bin_path = os.path.join(data_root, bin_file)
        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(bin_path), 'png')
        bin2img(bin_path, out_dir, height, width)


if __name__ == '__main__':
    decode_bin_files()
