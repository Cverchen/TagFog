 python train.py\
    --expriment_name 'cifar100_resnet18_supcon'\
    --model 'resnet18'\
    --dataset 'cifar100'\
    --size 32 \
    --method 'vision_text'\
    --epochs 200\
    --alpha 1\
    --beta 1\
    --batch_size 512\
    --test_batch_size 256\
    --num_workers 16\
    --learning_rate 0.05\
    --lr_decay_epochs '150'\
    --lr_decay_rate 0.1\
    --weight_decay 1e-4\
    --momentum 0.9\
    --jigsaw True\
    --temp_ci 0.1\
    --temp_sup 0.1\
    --save_freq 230\
    --print_freq 500\ 


        