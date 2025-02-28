cd /home/guantp/Infrared/MIRST/motive_target_gen
python3 motive_target_gen.py

mv /home/guantp/Infrared/MIRST/motive_target_gen/motive_target_imgs/images/* /home/guantp/Infrared/MIRST/DTUM/dataset/Tanzhe/images/1/
mv /home/guantp/Infrared/MIRST/motive_target_gen/motive_target_imgs/masks/* /home/guantp/Infrared/MIRST/DTUM/dataset/Tanzhe/masks/1/

cd /home/guantp/Infrared/MIRST/DTUM/dataset/Tanzhe/
python3 get_id.py
