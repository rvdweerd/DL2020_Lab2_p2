python train.py --seed 42 --txt_file ./assets/book_NL_tolstoy_anna_karenina.txt --batch_size 32 --train_steps=250 --seq_length 30 --print_every 1000 --sample_every 1000 --learning_rate 2e-3 --lstm_num_layers 2 --dropout_keep_prob 0.1 --learning_rate_decay=0.5