import getpass


class STATIC_VALUES:
    base_dir = 'D:\\Other\\Dataset\\voc2012\\' if getpass.getuser() == 'Reza' else 'D:\\Data\\voc2012\\'
    local_dir = base_dir + 'proc\\'
    raw_labels_dir = base_dir + 'labels\\'
    raw_images_dir = base_dir + 'images\\'
    raw_split_dir = base_dir + 'split\\'
    raw_test_dir = base_dir + 'test\\images\\'

    # local directories
    images_dir = local_dir + 'images\\'
    labels_file = local_dir + 'labels.npy'
    test_dir = local_dir + 'test\\'
    test_names_file = local_dir + 'test.txt'

    xception_graph_location = local_dir + 'graphs/xception'

    labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    labels_count = len(labels)

    image_size = [256, 256]

    batch_size = 5 if getpass.getuser() == 'Reza' else 20
    batches_in_memory = 100 if getpass.getuser() == 'Reza' else 150

    images_per_file = 15000
