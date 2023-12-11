 python train.py\
    --expriment_name 'imagenet100-ii_jigsaw'\
    --model 'resnet50'\
    --dataset 'ImageNet100-II'\
    --size 224 \
    --method 'vision_text'\
    --epochs 200\
    --alpha 1\
    --beta 1\
    --batch_size 128\
    --test_batch_size 128\
    --num_workers 16\
    --learning_rate 0.05\
    --lr_decay_epochs '100, 150, 180'\
    --lr_decay_rate 0.1\
    --weight_decay 1e-4\
    --momentum 0.9\
    --jigsaw True\
    --temp_ci 0.1\
    --temp_sup 0.1\
    --save_freq 50\
    --print_freq 5000\ 


        