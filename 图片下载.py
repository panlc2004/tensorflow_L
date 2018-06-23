import os
import sys
import urllib


def _progress(filename, count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' %
                     (filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()


def download(url):
    dest_directory = "image_download"
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    img_name = url.split("/")[-1]
    img_path = os.path.join(dest_directory, img_name)
    filepath, _ = urllib.request.urlretrieve(url, dest_directory, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', img_name, statinfo.st_size, 'bytes.')


def openFile(urlfile):
    with open(urlfile, 'r') as urlfile:
        for line in urlfile:
            download(line)

openFile("imagenet/test.txt")
