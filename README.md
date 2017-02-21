#YOLO_tensorflow

(Version 0.3, Last updated :2017.02.21)

###1.Introduction

This is tensorflow implementation of the YOLO:Real-Time Object Detection

It can only do predictions using pretrained YOLO_small & YOLO_tiny network for now.

(+ YOLO_face detector from https://github.com/quanhua92/darknet )

I extracted weight values from darknet's (.weight) files.

My code does not support training. Use darknet for training.

Original code(C implementation) & paper : http://pjreddie.com/darknet/yolo/

###2.Install
(1) Download code

(2) Download YOLO weight file from

YOLO_small : https://drive.google.com/file/d/0B2JbaJSrWLpza08yS2FSUnV2dlE/view?usp=sharing

YOLO_tiny  : https://drive.google.com/file/d/0B2JbaJSrWLpza0FtQlc3ejhMTTA/view?usp=sharing

YOLO_face : https://drive.google.com/file/d/0B2JbaJSrWLpzMzR5eURGN2dMTk0/view?usp=sharing

(3) Put the 'YOLO_(version).ckpt' in the 'weight' folder of downloaded code

###3.Usage

(1) direct usage with default settings (display on console, show output image, no output file writing)

	python YOLO_(small or tiny)_tf.py -fromfile (input image filename)

(2) direct usage with custom settings

	python YOLO_(small or tiny)_tf.py argvs

	where argvs are

	-fromfile (input image filename) : input image file
	-disp_console (0 or 1) : whether display results on terminal or not
	-imshow (0 or 1) : whether display result image or not
	-tofile_img (output image filename) : output image file
	-tofile_txt (output txt filename) : output text file (contains class, x, y, w, h, probability)

(3) import on other scripts

	import YOLO_(small or tiny)_tf
	yolo = YOLO_(small or tiny)_tf.YOLO_TF()

	yolo.disp_console = (True or False, default = True)
	yolo.imshow = (True or False, default = True)
	yolo.tofile_img = (output image filename)
	yolo.tofile_txt = (output txt filename)
	yolo.filewrite_img = (True or False, default = False)
	yolo.filewrite_txt = (True of False, default = False)

	yolo.detect_from_file(filename)
	yolo.detect_from_cvmat(cvmat)

###4.Requirements

- Tensorflow
- Opencv2

###5.Copyright

According to the LICENSE file of the original code, 
- Me and original author hold no liability for any damages
- Do not use this on commercial!

###6.Changelog
2016/02/15 : First upload!

2016/02/16 : Added YOLO_tiny, Fixed bug that ignores one of the boxes in grid when both boxes detected valid objects

2016/08/26 : Uploaded weight file converter! (darknet weight -> tensorflow ckpt)

2017/02/21 : Added YOLO_face (Thanks https://github.com/quanhua92/darknet)
