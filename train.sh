# commands to train with bin sizes [128, 256, 512, 1024, 2048] and mask lengths [10, 15, 20, 25, 30]
# accelerate launch training_mpm.py --bin_size 128 --mask_l 10
# accelerate launch training_mpm.py --bin_size 128 --mask_l 15
# accelerate launch training_mpm.py --bin_size 128 --mask_l 20
# accelerate launch training_mpm.py --bin_size 128 --mask_l 25
# accelerate launch training_mpm.py --bin_size 128 --mask_l 30
# accelerate launch training_mpm.py --bin_size 256 --mask_l 10
# accelerate launch training_mpm.py --bin_size 256 --mask_l 15
# accelerate launch training_mpm.py --bin_size 256 --mask_l 20
# accelerate launch training_mpm.py --bin_size 256 --mask_l 25
# accelerate launch training_mpm.py --bin_size 256 --mask_l 30
# accelerate launch training_mpm.py --bin_size 512 --mask_l 10
# accelerate launch training_mpm.py --bin_size 512 --mask_l 15
# accelerate launch training_mpm.py --bin_size 512 --mask_l 20
# accelerate launch training_mpm.py --bin_size 512 --mask_l 25
# accelerate launch training_mpm.py --bin_size 512 --mask_l 30
# accelerate launch training_mpm.py --bin_size 1024 --mask_l 10
# accelerate launch training_mpm.py --bin_size 1024 --mask_l 15
# accelerate launch training_mpm.py --bin_size 1024 --mask_l 20
# accelerate launch training_mpm.py --bin_size 1024 --mask_l 25
# accelerate launch training_mpm.py --bin_size 1024 --mask_l 30
# accelerate launch training_mpm.py --bin_size 2048 --mask_l 10
# accelerate launch training_mpm.py --bin_size 2048 --mask_l 15
# accelerate launch training_mpm.py --bin_size 2048 --mask_l 20
# accelerate launch training_mpm.py --bin_size 2048 --mask_l 25
# accelerate launch training_mpm.py --bin_size 2048 --mask_l 30
# commands to additionally train with bin sizes [32, 64] and mask lengths [2, 5]
accelerate launch training_mpm.py --bin_size 32 --mask_l 2
accelerate launch training_mpm.py --bin_size 32 --mask_l 5
accelerate launch training_mpm.py --bin_size 32 --mask_l 10
accelerate launch training_mpm.py --bin_size 32 --mask_l 15
accelerate launch training_mpm.py --bin_size 32 --mask_l 20
accelerate launch training_mpm.py --bin_size 32 --mask_l 25
accelerate launch training_mpm.py --bin_size 32 --mask_l 30
accelerate launch training_mpm.py --bin_size 64 --mask_l 2
accelerate launch training_mpm.py --bin_size 64 --mask_l 5
accelerate launch training_mpm.py --bin_size 64 --mask_l 10
accelerate launch training_mpm.py --bin_size 64 --mask_l 15
accelerate launch training_mpm.py --bin_size 64 --mask_l 20
accelerate launch training_mpm.py --bin_size 64 --mask_l 25
accelerate launch training_mpm.py --bin_size 64 --mask_l 30
accelerate launch training_mpm.py --bin_size 128 --mask_l 2
accelerate launch training_mpm.py --bin_size 128 --mask_l 5
accelerate launch training_mpm.py --bin_size 256 --mask_l 2
accelerate launch training_mpm.py --bin_size 256 --mask_l 5
accelerate launch training_mpm.py --bin_size 512 --mask_l 2
accelerate launch training_mpm.py --bin_size 512 --mask_l 5
accelerate launch training_mpm.py --bin_size 1024 --mask_l 2
accelerate launch training_mpm.py --bin_size 1024 --mask_l 5
accelerate launch training_mpm.py --bin_size 2048 --mask_l 2
accelerate launch training_mpm.py --bin_size 2048 --mask_l 5