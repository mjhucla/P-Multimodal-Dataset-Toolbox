from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cStringIO
import logging
import numpy as np
import multiprocessing
import os
from PIL import Image
import urllib2

logger = logging.getLogger('PyUtils')
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)4s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


def num_to_fix_length_str(num, l):
  num_str = str(num)
  return '0' * (l - len(num_str)) + num_str
  

def image_downloader(args, verbose=False, timeout=20):
  """Download an image from url.
  
  Args:
    args: a tuple contains three arguement.
        (1): local path of the image to save
        (2): image url
        (3): image_max_len. if not None, resizes the image with maximum length 
            and then crop the central part. The result image will be of size
            [image_max_len, image_max_len].
    verbose: whether to write the infomration to logger.
    timeout: url requested timeout (in seconds).
  
  Returns:
    1 for success, 0 for failure.
  """
  assert len(args) == 3 or len(args) == 2
  path = args[0]
  url = args[1]
  if len(args) == 3:
    image_max_len = args[2]
  else:
    image_max_len = None
  try:
    if verbose:
      logger.info('Downloading %s from %s', path, url)
    file_tmp = cStringIO.StringIO(urllib2.urlopen(url, timeout=timeout).read())
    img = Image.open(file_tmp)
    if image_max_len is not None:
      w, h = img.size
      if w >= h:
        w = int(image_max_len / h * w)
        h = image_max_len
        img = img.resize((w, h))
        s = (w - h) // 2
        img = img.crop((s, 0, s + h, h))
      else:
        h = int(image_max_len / w * h)
        w = image_max_len
        img = img.resize((w, h))
        s = (h - w) // 2
        img = img.crop((0, s, w, s + w))
    img.save(path)
    return 1
  except:
    if os.path.isfile(path):
      os.remove(path)
    return 0
  

class PinDataset(object):
  def __init__(self,
               meta_data_root='./pinterest_multimodal_2016_v1',
               name_prefix='pin_2016_v1',
               partitions=xrange(100)):
    """Initialize the Pinterest Multimodal Dataset.
    We will NOT load meta data to the memory because it will take ~6G
        memory. Instead, we save all the meta data paths.
    Explicit call load_meta_data() if you want to load them.
    
    Args:
      meta_data_root: root of meta data.
      name_prefix: prefix of the file name of the meta data files.
      partitions: index of the partitions to load.
    """
    self.annos = None
    self.pidx = None
    self.meta_data_root = meta_data_root
    self.meta_paths = []
    for ind in partitions:
      file_path = os.path.join(meta_data_root, 
                               self._get_meta_file_name(name_prefix, ind))
      self.meta_paths.append(file_path)
    logger.info('Pintrest Muldimodal Dataset initialized.')
      
  def download_images(self,
                      image_dir='./images',
                      ignore_existed=True,
                      image_max_len=224,
                      num_workers=8):
    """Download images in the dataset in parallel.
    
    Args:
      image_dir: directory of the images to save.
      ignore_existed: whether to ignore the existed images in the image_dir.
      image_max_len: if None, download and save the image with original size,
          otherwise, resize the image and crop the central part to the size of
          [image_max_len, image_max_len].
      num_workers: number of workers to parallel download the images.
      
    Returns:
      number of successfully downloaded images and the total images to download.
    """
    if not os.path.isdir(image_dir):
      os.makedirs(image_dir)
    
    meta_paths = self.meta_paths
    existed_images = set(os.listdir(image_dir)) if ignore_existed else set([])
    
    num_images = 0
    num_existed = 0
    num_download = 0
    
    for (ind_m, meta_path) in enumerate(meta_paths):
      annos = np.load(meta_path).tolist()
      list_path_url = []
      for anno in annos:
        image_name = anno['image_name']
        if image_name in existed_images:
          num_existed += 1
        else:
          num_images += 1
          list_path_url.append(
              (os.path.join(image_dir, image_name), anno['url'], image_max_len))
          
      pool = multiprocessing.Pool(num_workers)
      state = pool.map(image_downloader, list_path_url)
      num_download += sum(state)
          
      logger.info(
          'Successfully download %d/%d images for %d/%d meta files.',
          num_download, num_images, ind_m + 1, len(meta_paths))
    print('Successfully download %d/%d images (ignore %d existed ones.)' % (
          num_download, num_images, num_existed))
    
    return (num_download, num_images)
              
  def load_meta_data(self):
    """Load meta data into the memory."""
    annos = []
    pidxs = {}
    for meta_path in self.meta_paths:
      annos.extend(np.load(meta_path).tolist())
    print('Meta data loaded, contains %d images.' % len(annos))
    for (ind_a, anno) in enumerate(annos):
      pidx = self._get_pidx_from_image_name(anno['image_name'])
      assert not (pidx in pidxs), 'Duplicated pidx for different images!'
      pidxs[pidx] = ind_a
    print('Pinterest image index mapping created.')
    self.annos = annos
    self.pidxs = pidxs
  
  def get_pidxs(self, max_return_num=None):
    """Return the list of pinterest image idx in the meta data."""
    assert self.annos is not None, 'Load meta data first, call load_meta_data()'
    pidxs_list = list(self.pidxs.iterkeys())
    if max_return_num is not None:
      num = min(len(pidxs_list), max_return_num)
    else:
      num = len(pidxs_list)
    return pidxs_list[:num]
  
  def get_annotation(self,
                     image_dir='./images',
                     anno_id=None,
                     pin_id=None):
    """Return the annotation information.
    
    Args:
      image_dir: directory to find the downloaded images.
      anno_id: id of the annotation.
      pin_id: id of the pinterest image (last digits in the image name).
    
    Returns:
      the PIL image (if the image is downloaded) and its annotation.
    """
    assert self.annos is not None, 'Load meta data first, call load_meta_data()'
    assert anno_id is not None or pin_id is not None, ('Must specified anno_id'
        'or pin_id!')
    if anno_id is not None:
      anno = self.annos[anno_id]
    else:
      assert pin_id in self.pidxs, 'No pin_id %d in meta data!' % pin_id
      anno = self.annos[self.pidxs[pin_id]]
    image_path = os.path.join(image_dir, anno['image_name'])
    if os.path.isfile(image_path):
      img = Image.open(image_path)
    else:
      print('No image %s existed in dir %s, please download the images first by'
            'calling download_images()!' % (anno['image_name'], image_dir))
      img = None
    return (img, anno)
      
  def _get_meta_file_name(self, prefix, index, l=4):
    return prefix + '_' + num_to_fix_length_str(index, l) + '.npy'
    
  def _get_pidx_from_image_name(self, image_name):
    return int(image_name.rsplit('_')[-1].split('.')[0])
