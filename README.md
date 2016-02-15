#YOLO_tensorflow

(Version 0.1, Last updated :2016.02.15)

###1.Introduction

This is tensorflow implementation of the YOLO:Real-Time Object Detection

It can only do predictions using pretrained YOLO_small network for now.

I'm gonna support training later.

I extracted weight values from darknet's (.weight) files.

Original code(C implementation) & paper : http://pjreddie.com/darknet/yolo/

###2.Install
(1) Download code

(2) Download YOLO_small weight file from

https://drive.google.com/file/d/0B2JbaJSrWLpza08yS2FSUnV2dlE/view?usp=sharing

(3) Put the 'YOLO_small.ckpt' in the 'weight' folder of downloaded code

###3.Usage

(1) direct usage with default settings (display on console, show output image, no output file writing)

	python YOLO_small_tf.py -fromfile (input image filename)

(2) direct usage with custom settings

	python YOLO_small_tf.py argvs

	where argvs are

	-fromfile (input image filename) : input image file
	-disp_console (0 or 1) : whether display results on terminal or not
	-imshow (0 or 1) : whether display result image or not
	-tofile_img (output image filename) : output image file
	-tofile_txt (output txt filename) : output text file (contains class, x, y, w, h, probability)

(3) import on other scripts

	import YOLO_small_tf
	yolo = YOLO_small_tf.YOLO_TF()

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
