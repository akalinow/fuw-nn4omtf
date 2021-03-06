# -*- coding: utf-8 -*-
"""
    Copyright (C) 2018 Jacek Łysiak
    MIT License

    OMTF dataset command line tool object.
"""

import nn4omtf
import os
from nn4omtf import OMTFDataset
from .tool import OMTFTool
from .dataset_tool_config import ACTION, parser_config
from .utils import create_parser


class OMTFDatasetTool(OMTFTool):


    def __init__(self):
        handlers = [
            (ACTION.SHOW, OMTFDatasetTool._show),
            (ACTION.CREATE, OMTFDatasetTool._create),
            (ACTION.CONVERT, OMTFDatasetTool._convert),
        ]
        super().__init__(parser_config, "OMTF dataset tool", handlers)


    def _create(FLAGS):
        path = '.'
        if FLAGS.outdir is not None:
            path = FLAGS.outdir
            os.makedirs(FLAGS.outdir, exist_ok=True)

        transform = None
        if FLAGS.transform is not None:
            transform = tuple(FLAGS.transform)

        ds = OMTFDataset(FLAGS.files, FLAGS.train, FLAGS.valid, FLAGS.test, 
                transform=transform, treshold=FLAGS.treshold)
        ds.generate()
        ds_path = os.path.join(path, FLAGS.file_pref + '-dataset')
        ds_stat = os.path.join(path, FLAGS.file_pref + '-stats')
        ds.save_dataset(ds_path)
        ds.save_stats(ds_stat)


    def _show(FLAGS):
        OMTFDataset.show(FLAGS.file)


    def _convert(FLAGS):
        """
        Convert list of ROOT dataset files into Numpy dataset.
        """
        flist = FLAGS.data
        name = FLAGS.dest

        print("Importing ROOT's stuff...")
        # I know... Importing here is not beautiful but...
        nn4omtf.import_root_utils()
        print("Creating directory: " + dest)
        os.makedirs(dest, exist_ok=True)
        
        total = len(flist)
        cnt = 1
        time_start = time.time()
        time_last = time_start
        print("Starting conversion of {} files".format(total))
        for f in flist:
            print("> " + f)

        for f in flist:
            print("Converting {} of {}: {}".format(cnt, total, f))
            data = nn4omtf.root_utils.load_root_dict(f)
            name = data[1]['name']
            sign = data[1]['sign']
            pt_code = data[1]['val']
            x = nn4omtf.root_utils.root_to_numpy(name, sign, pt_code, data[0])
            path = os.path.join(dest, "{}_{}_{}.npz".format(name, sign, pt_code))
            print("Saving converted data as " + path)
            nn4omtf.utils.save_dict_as_npz(path, **x)
            now = time.time()
            print("Done in %.2f sec." % (now-time_last))
            print("Elapsed from start: {}' {}''".format(int((now - time_start) // 60), int(now - time_start) % 60))
            cnt += 1
            time_last = now

        print("Conversion finished!")

