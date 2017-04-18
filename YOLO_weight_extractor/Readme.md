# YOLO weight converter (darknet -> tensorflow)

1. Usage

   (1) download this modified version of darknet


   (2) put your darknet weight file(made by pjreddie or you) in the folder that contains darknet executable


   (3) run yolo in test mode (ex -> ./darknet yolo test cfg/yolo-small.cfg yolo-small.weights)


   (4) modified yolo will write txt files in folder 'cjy'


   (5) exit yolo when you see 'enter image path:'


   (6) open builder python file (YOLO_full_builder.py or YOLO_small_builder.py or YOLO_tiny_builder.py)


   (7) change weights_dir in line 6 (the folder that contains extracted txt files)


   (8) change path in the last line of function 'build_networks' (this is the path that will store ckpt file.)


   (9) run builder python script

2. Copyright

   
    I modified prejeddie's darknet code. (https://github.com/pjreddie/darknet)
