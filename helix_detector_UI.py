#!/usr/bin/env python
import sys

import dataset_manager
import model
import helix_prediction

if '--help' in sys.argv:
    print('Possible commands:\n'
          'python helix_detector_UI.py --help\n'
          'python helix_detector_UI.py -train  path_to_save_net dataset_path [evaluation_output]\n'
          'python helix_detector_UI.py -detect_helices net_path protein_mrc_path output_mrc_path\n')
#          'python helix_detector_UI.py -evaluate net_directory dataset_path evaluation_report_path\n')

if '-train' in sys.argv:
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        print('-train requires 4 arguments. run python helix_detector_UI.py --help')
        exit()
    dataset = list(dataset_manager.get_dataset(sys.argv[3]))[0:20]
    training_set, validation_set = dataset_manager.split_to_training_and_validation_sets(dataset)
    net = model.train_net(training_set, epochs=1, learning_rate=0.01)
    model.save_net(net, sys.argv[2])
    if len(sys.argv) == 5:
        #evaluation is requested
        model.evaluate(net,validation_set,save_to=sys.argv[4])

elif '-detect_helices' in sys.argv:
    if len(sys.argv) != 5:
        print('-detect_helices requires 5 arguments. run python helix_detector_UI.py --help')
        exit()
    helix_prediction.run_net_on_mrc_and_save(sys.argv[2], sys.argv[3], sys.argv[4])